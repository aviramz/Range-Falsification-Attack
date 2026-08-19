"""
animate.py
==========
Produces an animated view of the same run as simulate.py: drones flying in
formation, the falsified link highlighted once the attack starts, and a
marker showing when each drone independently declares the attacker
compromised (Sec. 6.4) -- all overlaid on the true (ground-truth)
trajectories, with the true minimum separation and collision threshold
tracked live in a side panel.

Run:
    python3 animate.py

Produces:
    outputs/swarm_animation.mp4  (if ffmpeg is available)
    outputs/swarm_animation.gif  (always -- no external dependency beyond Pillow)
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

import simulate as S

OUT = S.OUT
FRAME_STRIDE = 2       # use every Nth simulated cycle as an animation frame
TRAIL_CYCLES = 40       # how many past cycles of trail to draw per drone


def build_animation(res: dict) -> None:
    pos = res["hist_pos"]                      # (CYCLES, N, 2)
    min_sep = res["true_min_sep_series"]        # (CYCLES,)
    detect_cycle = res["detection_cycle_per_observer"]
    collision_cycle = res["collision_cycle"]

    # Truncate to the same physically-meaningful window used in Part 1
    # plots (see simulate.py: unbounded post-collision drift is a
    # controller-modeling artifact, not part of the phenomenon shown).
    cutoff = min(S.CYCLES, collision_cycle + 60) if collision_cycle is not None else S.CYCLES
    frame_cycles = list(range(0, cutoff, FRAME_STRIDE))

    labels = [f"D{k+1}" for k in range(S.N)]
    colors = ["#7f7f7f"] * S.N
    colors[S.ATTACKER] = "#d62728"
    colors[S.VICTIM] = "#1f77b4"

    fig, (ax_map, ax_sep) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})

    # ---- Map panel ----
    all_xy = pos[:cutoff].reshape(-1, 2)
    pad = 3.0
    ax_map.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax_map.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x (m)"); ax_map.set_ylabel("y (m)")
    title_map = ax_map.set_title("")

    trail_lines = [ax_map.plot([], [], color=colors[k], lw=1.0, alpha=0.5)[0] for k in range(S.N)]
    dots = [ax_map.plot([], [], "o", color=colors[k], ms=9, mec="black", mew=0.6, zorder=5)[0]
            for k in range(S.N)]
    labels_txt = [ax_map.text(0, 0, labels[k], fontsize=8, ha="center", va="bottom", zorder=6)
                  for k in range(S.N)]
    attack_link, = ax_map.plot([], [], color="#d62728", lw=2.0, ls=(0, (4, 2)), zorder=4)
    detect_flags = {i: ax_map.text(0, 0, "", fontsize=13, color="#2ca02c", ha="center", zorder=7)
                     for i in range(S.N) if i != S.ATTACKER}

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[S.ATTACKER],
               markeredgecolor="black", label=f"{labels[S.ATTACKER]} (attacker)", ms=9),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[S.VICTIM],
               markeredgecolor="black", label=f"{labels[S.VICTIM]} (victim)", ms=9),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f",
               markeredgecolor="black", label="other drones", ms=9),
        Line2D([0], [0], color="#d62728", lw=2.0, ls=(0, (4, 2)), label="falsified link (active)"),
        Line2D([0], [0], marker="$\\checkmark$", color="#2ca02c", lw=0, label="declared compromised", ms=10),
    ]
    ax_map.legend(handles=legend_elems, fontsize=7.5, loc="upper right")

    # ---- Separation panel ----
    t_full = np.arange(cutoff) * S.DT
    ax_sep.plot(t_full, min_sep[:cutoff], color="#333333", lw=1.3)
    ax_sep.axhline(S.D_MIN, color="#d62728", ls="--", lw=1.0, label=f"collision radius ({S.D_MIN} m)")
    ax_sep.axvline(S.ATTACK_ONSET_CYCLE * S.DT, color="#999999", ls=":", lw=1.0, label="attack onset")
    time_marker = ax_sep.axvline(0, color="#2ca02c", lw=1.5)
    ax_sep.set_xlabel("time (s)"); ax_sep.set_ylabel("true min. pairwise separation (m)")
    ax_sep.set_title("Physical safety margin (Part 1)")
    ax_sep.legend(fontsize=8, loc="upper right")

    fig.tight_layout()

    def init():
        for ln in trail_lines:
            ln.set_data([], [])
        for d in dots:
            d.set_data([], [])
        attack_link.set_data([], [])
        return trail_lines + dots + [attack_link]

    def frame_update(cycle):
        t = cycle * S.DT
        attack_active = cycle >= S.ATTACK_ONSET_CYCLE
        collided = collision_cycle is not None and cycle >= collision_cycle

        status = "ATTACK ACTIVE" if attack_active else "nominal"
        if collided:
            status += "  |  COLLISION"
        title_map.set_text(f"t = {t:5.2f}s   [{status}]")
        title_map.set_color("#d62728" if (attack_active or collided) else "black")

        trail_start = max(0, cycle - TRAIL_CYCLES * FRAME_STRIDE)
        for k in range(S.N):
            traj = pos[trail_start:cycle + 1, k, :]
            trail_lines[k].set_data(traj[:, 0], traj[:, 1])
            dots[k].set_data([pos[cycle, k, 0]], [pos[cycle, k, 1]])
            labels_txt[k].set_position((pos[cycle, k, 0], pos[cycle, k, 1] + 0.6))

        if attack_active:
            a_xy = pos[cycle, S.ATTACKER]
            v_xy = pos[cycle, S.VICTIM]
            attack_link.set_data([a_xy[0], v_xy[0]], [a_xy[1], v_xy[1]])
        else:
            attack_link.set_data([], [])

        for i, txt in detect_flags.items():
            dc = detect_cycle.get(i)
            if dc is not None and cycle >= dc:
                txt.set_position((pos[cycle, i, 0], pos[cycle, i, 1] - 0.9))
                txt.set_text("\u2713 flagged D2")
            else:
                txt.set_text("")

        time_marker.set_xdata([t, t])

        return trail_lines + dots + labels_txt + [attack_link, title_map, time_marker] + list(detect_flags.values())

    anim = animation.FuncAnimation(fig, frame_update, frames=frame_cycles, init_func=init,
                                    interval=1000 / 20, blit=False)

    gif_path = os.path.join(OUT, "swarm_animation.gif")
    anim.save(gif_path, writer=animation.PillowWriter(fps=20))
    print(f"Wrote {gif_path}")

    try:
        mp4_path = os.path.join(OUT, "swarm_animation.mp4")
        anim.save(mp4_path, writer=animation.FFMpegWriter(fps=20, bitrate=1800))
        print(f"Wrote {mp4_path}")
    except Exception as e:
        print(f"Skipped MP4 (ffmpeg not available or failed): {e}")

    plt.close(fig)


if __name__ == "__main__":
    result = S.run()
    build_animation(result)
