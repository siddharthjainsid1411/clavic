"""
Compare tau=2s (saved checkpoint) vs tau=0.5s (freshly optimized) trajectories.

Two output figures:
  Fig 1 — 2D workspace: start, goal, human, comfort zone, both trajectories
  Fig 2 — Time-series grid:
             Row 1: X, Y, Z  position
             Row 2: X, Y, Z  velocity
             Row 3: K[0,0], K[1,1], K[2,2]  (diagonal stiffness)
"""

import numpy as np
import matplotlib.pyplot as plt

from core.certified_policy import CertifiedPolicy
from spec.compiler import Compiler
from spec.json_parser import load_taskspec_from_json
from logic.predicates import at_goal_pose, human_comfort_distance, velocity_limit, human_body_exclusion
from optimization.optimizer import PI2


# ── Scene constants (same as certified_policy.py) ──────────────────────────
START_POS    = np.array([0.55, 0.00, 0.11])
GOAL_POS     = np.array([0.05, 0.72, 0.11])
HUMAN_POS    = np.array([0.30, 0.40, 0.11])
COMFORT_DIST = 0.15
BODY_RADIUS  = 0.06   # hard exclusion — physical body of human
GOAL_TOL     = 0.05
VMAX         = 0.5

C5 = "#9b59b6"   # purple  — tau=2s saved
C3 = "#e74c3c"   # red     — tau=0.5s optimized




# ── Step 1: Load saved 5s trajectory ───────────────────────────────────────
def load_5s_checkpoint(path="optimal_checkpoint.npz"):
    ckpt = np.load(path)
    tau  = float(ckpt["horizon_sec"])
    print(f"✓ Loaded checkpoint  tau={tau:.1f}s  cost={float(ckpt['best_cost']):.4f}")
    return {
        "position": ckpt["position"],
        "velocity": ckpt["velocity"],
        "time":     ckpt["time"],
        "K":        ckpt["K_trace"],    # (T,3,3)
        "tau":      tau,
        "cost":     float(ckpt["best_cost"]),
    }


# ── Step 2: Run PI2 optimization at tau=0.5s ──────────────────────────────
def optimize_tau_short():
    tau = 0.5
    print(f"\nRunning PI2 optimization at tau={tau}s ...")

    taskspec = load_taskspec_from_json("spec/example_task.json")
    predicate_registry = {
        "AtGoalPose":            at_goal_pose,
        "HumanComfortDistance":  human_comfort_distance,
        "HumanBodyExclusion":    human_body_exclusion,
        "VelocityLimit":         velocity_limit,
    }
    objective_fn = Compiler(predicate_registry).compile(taskspec)

    policy    = CertifiedPolicy(tau=tau)
    theta_dim = policy.parameter_dimension()

    theta_init = np.zeros(theta_dim)
    sigma_init = np.ones(theta_dim) * 5.0
    pi2 = PI2(theta=theta_init, sigma=sigma_init, lam=0.01, decay=0.98)

    N_SAMPLES, N_UPDATES = 12, 80
    best_cost  = float("inf")
    best_theta = theta_init.copy()

    for i in range(N_UPDATES):
        samples = pi2.sample(N_SAMPLES)
        costs   = np.array([objective_fn(policy.rollout(s)) for s in samples])
        pi2.update(samples, costs)

        if costs.min() < best_cost:
            best_cost  = costs.min()
            best_theta = samples[np.argmin(costs)].copy()

        print(f"  Update {i+1:02d} | Min: {costs.min():.4f} | Mean: {costs.mean():.4f} | Best: {best_cost:.4f}")

    trace = policy.rollout(best_theta)
    print(f"✓ tau={tau}s optimization done  cost={best_cost:.4f}")
    return {
        "position": trace.position,
        "velocity": trace.velocity,
        "time":     trace.time,
        "K":        trace.gains["K"],   # (T,3,3)
        "tau":      tau,
        "cost":     best_cost,
    }


# ── Figure 1: 2D Workspace ──────────────────────────────────────────────────
def plot_workspace(traj_5s, traj_3s):
    fig, ax = plt.subplots(figsize=(5, 6))

    # Direct path — dashed gray faded line from start to goal
    ax.plot([START_POS[0], GOAL_POS[0]], [START_POS[1], GOAL_POS[1]],
            color="gray", linewidth=0.8, linestyle="--", alpha=0.4, zorder=1, label="Shortest path")

    # Comfort zone fill + ring
    ax.add_patch(plt.Circle(HUMAN_POS[:2], COMFORT_DIST,
                            color="orange", fill=True, alpha=0.12, zorder=1))
    ax.add_patch(plt.Circle(HUMAN_POS[:2], COMFORT_DIST,
                            color="orange", fill=False, linestyle="--", linewidth=0.8, alpha=0.7))

    # Body exclusion
    ax.add_patch(plt.Circle(HUMAN_POS[:2], BODY_RADIUS,
                            color="red", fill=True, alpha=0.35, zorder=3))
    ax.add_patch(plt.Circle(HUMAN_POS[:2], BODY_RADIUS,
                            color="red", fill=False, linestyle="-", linewidth=0.8, zorder=3))

    # Legend proxy patches for the two human zones
    from matplotlib.patches import Patch
    legend_comfort = Patch(facecolor="orange", alpha=0.4, edgecolor="orange",
                           linestyle="--", linewidth=0.8, label=f"Comfort region (r={COMFORT_DIST}m)")
    legend_body    = Patch(facecolor="red",    alpha=0.5, edgecolor="red",
                           linestyle="-",  linewidth=0.8, label=f"Body exclusion (r={BODY_RADIUS}m)")

    # Goal tolerance ring
    ax.add_patch(plt.Circle(GOAL_POS[:2], GOAL_TOL,
                            color="black", fill=False, linestyle=":", linewidth=0.8, alpha=0.5))

    # Trajectories — publication quality linewidth
    ax.plot(traj_5s["position"][:, 0], traj_5s["position"][:, 1],
            color=C5, linewidth=1.5, label=f"Time = {traj_5s['tau']:.1f}s", zorder=4)
    ax.plot(traj_3s["position"][:, 0], traj_3s["position"][:, 1],
            color=C3, linewidth=1.5, linestyle="--", label=f"Time = {traj_3s['tau']:.1f}s", zorder=4)
    
    # Extend trajectories to goal with same style (showing completion)
    ax.plot([traj_5s["position"][-1, 0], GOAL_POS[0]], [traj_5s["position"][-1, 1], GOAL_POS[1]],
            color=C5, linewidth=1.5, linestyle="--", alpha=0.5, zorder=3)
    ax.plot([traj_3s["position"][-1, 0], GOAL_POS[0]], [traj_3s["position"][-1, 1], GOAL_POS[1]],
            color=C3, linewidth=1.5, linestyle="--", alpha=0.5, zorder=3)

    # Start / Goal / Human markers
    ax.plot(*START_POS[:2], "o", color="limegreen", markersize=8,
            markeredgecolor="black", markeredgewidth=0.9, label="Start", zorder=5)
    ax.plot(*GOAL_POS[:2], "o", color="black", markersize=8,
            markeredgecolor="black", markeredgewidth=0.9, label="Goal", zorder=5)
    ax.plot(*HUMAN_POS[:2], "o", color="orange", markersize=8,
            markeredgecolor="black", markeredgewidth=0.9, label="Human", zorder=5)

    ax.set_xlim(-0.05, 0.75)
    ax.set_ylim(-0.15, 0.90)
    ax.set_xlabel("X (m)", fontsize=10, fontweight="normal")
    ax.set_ylabel("Y (m)", fontsize=10, fontweight="normal")
    ax.tick_params(labelsize=9)
    handles, labels = ax.get_legend_handles_labels()
    handles += [legend_comfort, legend_body]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.92,
              edgecolor="black", handlelength=1.6, borderpad=0.7,
              labelspacing=0.5, handletextpad=0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    plt.savefig("comparison_workspace.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("✓ Saved  →  comparison_workspace.png")


# ── Figure 2: Time-series grid ──────────────────────────────────────────────
def plot_timeseries(traj_5s, traj_3s):
    """
    3 rows × 3 cols:
      Row 0: position  X, Y, Z
      Row 1: velocity  X, Y, Z
      Row 2: stiffness K[0,0], K[1,1], K[2,2]
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))

    lbl_5s = f"Time = {traj_5s['tau']:.1f}s"
    lbl_3s = f"Time = {traj_3s['tau']:.1f}s"

    labels_pos  = ["X  (m)",      "Y  (m)",      "Z  (m)"]
    labels_vel  = ["Vx  (m/s)",   "Vy  (m/s)",   "Vz  (m/s)"]
    labels_k    = ["K[0,0]  (N/m)", "K[1,1]  (N/m)", "K[2,2]  (N/m)"]
    goal_vals   = GOAL_POS
    start_vals  = START_POS

    for col in range(3):

        # ── Row 0: Position ──────────────────────────────────────────────
        ax = axes[0, col]
        ax.plot(traj_5s["time"], traj_5s["position"][:, col],
                color=C5, linewidth=2.0, label=lbl_5s)
        ax.plot(traj_3s["time"], traj_3s["position"][:, col],
                color=C3, linewidth=2.0, linestyle="--", label=lbl_3s)
        ax.axhline(goal_vals[col],  color="black", linestyle=":", linewidth=1.0,
                   alpha=0.6, label=f"goal={goal_vals[col]:.2f}")
        ax.axhline(start_vals[col], color="green", linestyle=":", linewidth=1.0,
                   alpha=0.6, label=f"start={start_vals[col]:.2f}")
        ax.set_ylabel(labels_pos[col], fontsize=9, fontweight="normal")
        ax.set_xlabel("t (s)", fontsize=9, fontweight="normal")
        ax.set_title(f"Position {['X','Y','Z'][col]}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.9, edgecolor="black", loc="best")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

        # ── Row 1: Velocity ──────────────────────────────────────────────
        ax = axes[1, col]
        ax.plot(traj_5s["time"], traj_5s["velocity"][:, col],
                color=C5, linewidth=2.0, label=lbl_5s)
        ax.plot(traj_3s["time"], traj_3s["velocity"][:, col],
                color=C3, linewidth=2.0, linestyle="--", label=lbl_3s)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(labels_vel[col], fontsize=9, fontweight="normal")
        ax.set_xlabel("t (s)", fontsize=9, fontweight="normal")
        ax.set_title(f"Velocity {['X','Y','Z'][col]}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.9, edgecolor="black", loc="best")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

        # ── Row 2: Stiffness K diagonal ──────────────────────────────────
        ax = axes[2, col]
        ax.plot(traj_5s["time"], traj_5s["K"][:, col, col],
                color=C5, linewidth=2.0, label=lbl_5s)
        ax.plot(traj_3s["time"], traj_3s["K"][:, col, col],
                color=C3, linewidth=2.0, linestyle="--", label=lbl_3s)
        ax.set_ylabel(labels_k[col], fontsize=9, fontweight="normal")
        ax.set_xlabel("t (s)", fontsize=9, fontweight="normal")
        ax.set_title(f"Stiffness K[{col},{col}]", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.9, edgecolor="black", loc="best")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

    fig.tight_layout(pad=0.8)
    plt.savefig("comparison_timeseries.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("✓ Saved  →  comparison_timeseries.png")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    traj_5s = load_5s_checkpoint()
    traj_3s = optimize_tau_short()
    plot_workspace(traj_5s, traj_3s)
    plot_timeseries(traj_5s, traj_3s)
