"""
compare_n.py
============
NOTE: this script was built when physical collision danger was still part
of what this simulation demonstrated (see README history). That is no
longer the focus -- the simulation now targets detection and attribution
only (Sec. 6), with random/multi-victim attacks (see simulate.py). This
script is kept for the N-scaling comparison it still validly shows (the
detection architecture needs zero changes to scale from 6 to 15 drones),
but its per-N attack tuning is calibrated for the collision outcome it was
originally built to demonstrate, not for the current default scenario.

Runs the same attack scenario at N=6 and N=15 side by side, into separate
output folders, to show the "dilution" effect discovered while scaling this
simulation up: each drone's formation guidance (Sec. 5.2) sums a spring
correction from every OTHER drone, so the same single falsified link is a
much smaller fraction of the total corrective signal at N=15 (1 lie among
14 neighbors) than at N=6 (1 lie among 5). With an unchanged attack rate,
N=15 shows no collision at all; ATTACK_STEP_SIZE_M is re-tuned per N below
to produce comparable physical danger, found by direct sweep (see README).

Run:
    python3 compare_n.py

Produces:
    outputs/n06/part1_*.png, part2_*.png, summary.txt
    outputs/n15/part1_*.png, part2_*.png, summary.txt
    outputs/comparison_summary.txt
"""
from __future__ import annotations
import os
import numpy as np
import simulate as S

BASE_OUT = S.OUT

CONFIGS = [
    {"n": 6, "step_size_m": 0.5, "label": "n06"},
    {"n": 15, "step_size_m": 1.2, "label": "n15"},
]


def run_one(n: int, step_size_m: float, label: str) -> dict:
    S.N = n
    S.INITIAL_POS = S.generate_formation(n, spacing=3.0)
    S.ATTACKER = 1
    dists = np.linalg.norm(S.INITIAL_POS - S.INITIAL_POS[S.ATTACKER], axis=1)
    dists[S.ATTACKER] = np.inf
    S.VICTIM = int(np.argmin(dists))
    S.VICTIMS = [S.VICTIM]  # single-victim comparison; AttackConfig reads VICTIMS
    S.ATTACK_STEP_SIZE_M = step_size_m
    S.ATTACK_MAGNITUDE_M = 50.0  # non-binding ceiling, see ranging.py docstring

    out_dir = os.path.join(BASE_OUT, label)
    os.makedirs(out_dir, exist_ok=True)
    S.OUT = out_dir

    print(f"\n=== N={n} (attacker=D{S.ATTACKER+1}, victim=D{S.VICTIM+1}, "
          f"step_size={step_size_m}m) ===")
    result = S.run()
    S.make_plots(result)
    S.write_summary(result)
    return result


def main():
    results = {}
    for cfg in CONFIGS:
        results[cfg["n"]] = run_one(cfg["n"], cfg["step_size_m"], cfg["label"])

    lines = ["=== N=6 vs N=15: formation-dilution comparison ===", ""]
    for cfg in CONFIGS:
        n = cfg["n"]
        res = results[n]
        coll = res["collision_cycle"]
        min_sep = res["true_min_sep_series"].min()
        lines.append(f"N={n} (step_size={cfg['step_size_m']}m):")
        if coll is not None:
            lines.append(f"  Collision at cycle {coll} (t={coll*S.DT:.2f}s), "
                          f"min separation reached: {min_sep:.2f}m")
        else:
            lines.append(f"  No collision; min separation reached: {min_sep:.2f}m")
        lines.append("")

    lines.append(
        "Takeaway: the same detection/attribution architecture (Sec. 6) scales\n"
        "cleanly to more drones with no structural changes. Physical danger from\n"
        "a FIXED attack rate does not scale the same way -- larger, more\n"
        "connected swarms are more physically robust to a single falsified link\n"
        "purely as a byproduct of full-mesh formation control (more honest\n"
        "neighbors diluting one liar's influence on the total corrective signal).\n"
        "This is a property of the control architecture, not of detection."
    )
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(BASE_OUT, "comparison_summary.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
