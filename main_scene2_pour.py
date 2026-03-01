"""
Scene 2 — Coffee Mug Pour Task (with Orientation DMP)

Task:
  Phase 1 (0-4 s):  Carry mug upright from start to goal (avoid obstacle)
  Phase 2 (4-6 s):  Hold at goal, keep upright (don't pour yet)
  Phase 3 (6-10 s): Pour — tilt mug 90° about Y axis

Key constraint: "Don't pour until you reach the goal"
  → Orientation stays upright during phases 1-2 (OrientationLimit)
  → Orientation reaches pour angle during phase 3 (OrientationAtTarget)

Plots generated:
  1. scene2_pour_workspace.png  — XY workspace with trajectory and obstacles
  2. scene2_pour_trace.png      — tr(K), tr(D), speed, orientation vs time
  3. scene2_pour_peraxis.png    — Per-axis Vx,Vy,Vz and Kxx,Kyy,Kzz
  4. scene2_pour_orientation.png — Quaternion and angular velocity evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from spec.compiler import Compiler
from spec.json_parser import load_taskspec_from_json
from core.multi_phase_policy import MultiPhaseCertifiedPolicy
from optimization.optimizer import PIBB
from experiment_checkpoint_warmstart import save_checkpoint
from core.cgms.quat_utils import quat_distance, quat_normalize

from logic.predicates import (
    at_goal_pose, at_waypoint, hold_at_waypoint,
    obstacle_avoidance, velocity_limit,
    orientation_at_target, orientation_limit,
    angular_velocity_limit,
)

# ── Seaborn styling ──────────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_palette("deep")

# ── Scene constants ──────────────────────────────────────────────────
START    = np.array([0.55, 0.00, 0.30])
GOAL     = np.array([0.30, 0.55, 0.30])
OBSTACLE = np.array([0.40, 0.30, 0.30])
OBS_RAD  = 0.10

Q_UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])
Q_POUR    = quat_normalize(np.array([0.70710678, 0.0, 0.70710678, 0.0]))

T_HOLD_START = 4.0
T_HOLD_END   = 6.0
T_POUR_END   = 10.0


# ── Predicate registry ───────────────────────────────────────────────
def build_predicate_registry():
    return {
        "AtGoalPose":            at_goal_pose,
        "AtWaypoint":            at_waypoint,
        "HoldAtWaypoint":        hold_at_waypoint,
        "ObstacleAvoidance":     obstacle_avoidance,
        "VelocityLimit":         velocity_limit,
        "OrientationAtTarget":   orientation_at_target,
        "OrientationLimit":      orientation_limit,
        "AngularVelocityLimit":  angular_velocity_limit,
    }


# ── Diagnostics ──────────────────────────────────────────────────────
def print_diagnostics(trace, best_cost):
    pos   = trace.position
    speed = np.linalg.norm(trace.velocity, axis=1)
    K_arr = trace.gains["K"]
    D_arr = trace.gains["D"]
    trK   = np.array([np.trace(K) for K in K_arr])
    trD   = np.array([np.trace(D) for D in D_arr])

    # Position checks
    d_goal = np.linalg.norm(pos - GOAL, axis=1)
    goal_reached = np.any(d_goal < 0.05)
    goal_t = trace.time[np.argmax(d_goal < 0.05)] if goal_reached else -1

    d_obs = np.linalg.norm(pos[:, :2] - OBSTACLE[:2], axis=1)
    obs_clear = (d_obs.min() - OBS_RAD) * 100  # cm

    # Impedance checks
    K_eig_min = min(np.linalg.eigvalsh(K)[0] for K in K_arr)
    D_eig_min = min(np.linalg.eigvalsh(D)[0] for D in D_arr)

    print(f"\n{'='*40} RESULT DIAGNOSTICS {'='*40}")
    print(f"  Best cost       : {best_cost:.4f}")
    print(f"  Goal reached    : {'YES' if goal_reached else 'NO'} at t={goal_t:.2f}s")
    print(f"  Max speed       : {speed.max():.4f} m/s  (limit 0.8 m/s)")
    print(f"  Obstacle clear  : {obs_clear:.1f} cm")
    print(f"  tr(K) range     : [{trK.min():.1f}, {trK.max():.1f}] N/m")
    print(f"  tr(D) range     : [{trD.min():.3f}, {trD.max():.3f}] Ns/m")
    print(f"  K eig min       : {K_eig_min:.4f}  (must be > 0)")
    print(f"  D eig min       : {D_eig_min:.4f}  (must be > 0)")

    # Orientation checks
    if trace.orientation is not None:
        # Check upright during carry/hold (phase 1-2)
        mask_carry = trace.time <= T_HOLD_END
        max_tilt_carry = max(
            quat_distance(trace.orientation[k], Q_UPRIGHT)
            for k in range(len(trace.time)) if mask_carry[k]
        )
        # Check pour angle at end
        d_pour_end = quat_distance(trace.orientation[-1], Q_POUR)

        # Angular velocity
        omega_norm = np.linalg.norm(trace.angular_velocity, axis=1)

        print(f"  Max tilt carry  : {np.degrees(max_tilt_carry):.1f} deg  (limit 15 deg)")
        print(f"  Pour error end  : {np.degrees(d_pour_end):.1f} deg  (target < 10 deg)")
        print(f"  Max omega       : {omega_norm.max():.3f} rad/s  (limit 1.5 rad/s)")
    print(f"{'='*100}\n")


# ── Plot 1: Workspace ────────────────────────────────────────────────
def plot_workspace(trace, best_cost, save_path="scene2_pour_workspace.png"):
    pos = trace.position

    fig, ax = plt.subplots(figsize=(13, 13))
    fig.suptitle(
        f"Scene 2: Coffee Pour — Workspace (XY plane, z={START[2]} m)\nbest_cost={best_cost:.3f}",
        fontsize=14, fontweight='bold'
    )

    # Obstacle exclusion circle
    circ = plt.Circle(OBSTACLE[:2], OBS_RAD,
                      color='red', fill=True, alpha=0.15,
                      label=f'Obstacle (r={OBS_RAD} m)')
    ax.add_patch(circ)
    ax.add_patch(plt.Circle(OBSTACLE[:2], OBS_RAD,
                            color='red', fill=False, linestyle='--', linewidth=2.0))

    # Goal tolerance circle
    ax.add_patch(plt.Circle(GOAL[:2], 0.04, color='blue', fill=False,
                            linestyle=':', linewidth=2.0, label='Goal tolerance (4 cm)'))

    # Phase-coloured trajectory
    t_p1_end = np.searchsorted(trace.time, T_HOLD_START)
    t_p2_end = np.searchsorted(trace.time, T_HOLD_END)
    ax.plot(pos[:t_p1_end+1, 0], pos[:t_p1_end+1, 1],
            'b-', lw=3, label='Phase 1: carry', zorder=3)
    ax.plot(pos[t_p1_end:t_p2_end+1, 0], pos[t_p1_end:t_p2_end+1, 1],
            'g-', lw=4, label='Phase 2: hold', zorder=4)
    ax.plot(pos[t_p2_end:, 0], pos[t_p2_end:, 1],
            'r-', lw=3, label='Phase 3: pour', zorder=3)

    # Key points
    ax.scatter(*START[:2], s=200, c='green', zorder=6, marker='o',
              edgecolor='darkgreen', linewidth=2, label='Start (mug)')
    ax.scatter(*GOAL[:2], s=350, c='black', zorder=6, marker='*',
              label='Goal (pour target)')
    ax.scatter(*OBSTACLE[:2], s=100, c='red', zorder=5, marker='+',
              linewidth=3, label='Obstacle center')

    ax.set_xlabel("X (m)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Y (m)", fontsize=12, fontweight='bold')
    ax.set_title("End-Effector Trajectory in 2D Workspace", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95, edgecolor='black',
              fancybox=False, shadow=False)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved -> {save_path}")
    plt.close()


# ── Plot 2: Trace metrics ────────────────────────────────────────────
def plot_trace_metrics(trace, best_cost, save_path="scene2_pour_trace.png"):
    speed = np.linalg.norm(trace.velocity, axis=1)
    K_arr = trace.gains["K"]
    D_arr = trace.gains["D"]
    trK   = np.array([np.trace(K) for K in K_arr])
    trD   = np.array([np.trace(D) for D in D_arr])

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    fig.suptitle(
        f"Scene 2: Coffee Pour — Impedance & Speed\nbest_cost={best_cost:.3f}",
        fontsize=12, fontweight='bold'
    )

    # Speed
    ax = axes[0]
    ax.plot(trace.time, speed, 'b-', lw=2.5, label='EE speed', zorder=2)
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.15, color='green', label='Hold phase', zorder=1)
    ax.axvspan(T_HOLD_END, T_POUR_END, alpha=0.10, color='orange', label='Pour phase', zorder=1)
    ax.axhline(0.8, color='red', linestyle='--', lw=1.5, alpha=0.7, label='Speed limit')
    ax.set_ylabel("Speed (m/s)", fontsize=10, fontweight='bold')
    ax.set_title("End-Effector Speed", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 10)

    # tr(K)
    ax = axes[1]
    ax.plot(trace.time, trK, 'm-', lw=2.5, label='tr(K)', zorder=2)
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.15, color='green', label='Hold', zorder=1)
    ax.axvspan(T_HOLD_END, T_POUR_END, alpha=0.10, color='orange', label='Pour', zorder=1)
    ax.set_ylabel("tr(K) (N/m)", fontsize=10, fontweight='bold')
    ax.set_title("Stiffness Evolution", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 10)

    # tr(D)
    ax = axes[2]
    ax.plot(trace.time, trD, 'c-', lw=2.5, label='tr(D)', zorder=2)
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.15, color='green', label='Hold', zorder=1)
    ax.axvspan(T_HOLD_END, T_POUR_END, alpha=0.10, color='orange', label='Pour', zorder=1)
    ax.axhline(30.0, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='Min safe tr(D)=30')
    ax.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
    ax.set_ylabel("tr(D) (Ns/m)", fontsize=10, fontweight='bold')
    ax.set_title("Damping Evolution", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved -> {save_path}")
    plt.close()


# ── Plot 3: Orientation evolution ─────────────────────────────────────
def plot_orientation(trace, best_cost, save_path="scene2_pour_orientation.png"):
    if trace.orientation is None:
        print("No orientation data — skipping orientation plot.")
        return

    q = trace.orientation        # (T, 4)
    omega = trace.angular_velocity  # (T, 3)
    t = trace.time

    # Compute geodesic distance to upright and to pour
    d_upright = np.array([quat_distance(q[k], Q_UPRIGHT) for k in range(len(t))])
    d_pour    = np.array([quat_distance(q[k], Q_POUR) for k in range(len(t))])
    omega_norm = np.linalg.norm(omega, axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(13, 14))
    fig.suptitle(
        f"Scene 2: Coffee Pour — Orientation Evolution\nbest_cost={best_cost:.3f}",
        fontsize=13, fontweight='bold'
    )

    # Row 0: Quaternion components
    ax = axes[0]
    labels_q = ['w', 'x', 'y', 'z']
    colors_q = sns.color_palette("deep", 4)
    for i, (lbl, c) in enumerate(zip(labels_q, colors_q)):
        ax.plot(t, q[:, i], color=c, lw=2.0, label=f'q_{lbl}')
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.12, color='green', label='Hold')
    ax.axvspan(T_HOLD_END, T_POUR_END, alpha=0.08, color='orange', label='Pour')
    ax.set_ylabel("Quaternion", fontsize=10, fontweight='bold')
    ax.set_title("Quaternion Components", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='black', ncol=3)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 10)

    # Row 1: Geodesic distances
    ax = axes[1]
    ax.plot(t, np.degrees(d_upright), 'b-', lw=2.5, label='Tilt from upright')
    ax.plot(t, np.degrees(d_pour), 'r-', lw=2.5, label='Distance to pour')
    ax.axhline(15, color='blue', linestyle='--', lw=1.5, alpha=0.6, label='Upright limit (15°)')
    ax.axhline(10, color='red', linestyle='--', lw=1.5, alpha=0.6, label='Pour tolerance (10°)')
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.12, color='green')
    ax.axvspan(T_HOLD_END, T_POUR_END, alpha=0.08, color='orange')
    ax.set_ylabel("Angle (deg)", fontsize=10, fontweight='bold')
    ax.set_title("Geodesic Distances", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='center right', framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 10)

    # Row 2: Angular velocity components
    ax = axes[2]
    labels_w = ['ωx', 'ωy', 'ωz']
    colors_w = sns.color_palette("Set2", 3)
    for i, (lbl, c) in enumerate(zip(labels_w, colors_w)):
        ax.plot(t, omega[:, i], color=c, lw=2.0, label=lbl)
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.12, color='green')
    ax.axvspan(T_HOLD_END, T_POUR_END, alpha=0.08, color='orange')
    ax.set_ylabel("ω (rad/s)", fontsize=10, fontweight='bold')
    ax.set_title("Angular Velocity Components", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 10)

    # Row 3: Angular velocity norm
    ax = axes[3]
    ax.plot(t, omega_norm, 'purple', lw=2.5, label='||ω||')
    ax.axhline(1.5, color='red', linestyle='--', lw=1.5, alpha=0.7, label='ω limit (1.5 rad/s)')
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.12, color='green')
    ax.axvspan(T_HOLD_END, T_POUR_END, alpha=0.08, color='orange')
    ax.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
    ax.set_ylabel("||ω|| (rad/s)", fontsize=10, fontweight='bold')
    ax.set_title("Angular Velocity Magnitude", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved -> {save_path}")
    plt.close()


# ── Plot 4: Per-axis ──────────────────────────────────────────────────
def plot_peraxis(trace, best_cost, save_path="scene2_pour_peraxis.png"):
    vel    = trace.velocity
    K_arr  = trace.gains["K"]
    K_diag = np.array([np.diag(K) for K in K_arr])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Scene 2: Coffee Pour — Per-Axis\nbest_cost={best_cost:.3f}",
        fontsize=12, fontweight='bold'
    )
    axis_labels = ['X', 'Y', 'Z']
    vel_colors  = ['steelblue', 'darkorange', 'seagreen']
    k_colors    = ['mediumpurple', 'tomato', 'goldenrod']

    for i, (lbl, vc, kc) in enumerate(zip(axis_labels, vel_colors, k_colors)):
        ax = axes[0, i]
        ax.plot(trace.time, vel[:, i], color=vc, lw=2.5, label=f'V{lbl}', zorder=2)
        ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.15, color='green', zorder=1)
        ax.axvspan(T_HOLD_END, T_POUR_END, alpha=0.08, color='orange', zorder=1)
        ax.set_ylabel(f"V{lbl} (m/s)", fontsize=10, fontweight='bold')
        ax.set_title(f"Velocity {lbl}-axis", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 10)

        ax = axes[1, i]
        ax.plot(trace.time, K_diag[:, i], color=kc, lw=2.5, label=f'K{lbl}{lbl}', zorder=2)
        ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.15, color='green', zorder=1)
        ax.axvspan(T_HOLD_END, T_POUR_END, alpha=0.08, color='orange', zorder=1)
        ax.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
        ax.set_ylabel(f"K{lbl}{lbl} (N/m)", fontsize=10, fontweight='bold')
        ax.set_title(f"Stiffness K_{lbl}{lbl}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved -> {save_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────
def main():
    taskspec = load_taskspec_from_json("spec/scene2_pour_task.json")
    assert taskspec.phases is not None, "scene2_pour_task.json must contain 'phases'"

    policy    = MultiPhaseCertifiedPolicy(taskspec.phases)
    theta_dim = policy.parameter_dimension()
    print(f"Multi-phase policy: {len(taskspec.phases)} phases, theta_dim = {theta_dim}")
    print(f"  has_orientation: {policy.has_orientation}")
    for idx, p in enumerate(taskspec.phases):
        ori_dim = policy.ori_dims[idx]
        print(f"  Phase {idx+1} ({p['label']}): pos_dim={policy.theta_dims[idx]-ori_dim}, ori_dim={ori_dim}")

    predicate_registry = build_predicate_registry()
    compiler     = Compiler(predicate_registry)
    objective_fn = compiler.compile(taskspec)

    # Nominal diagnostics
    trace0 = policy.rollout(np.zeros(theta_dim))
    cost0  = objective_fn(trace0)
    print(f"Nominal (theta=0) — start: {trace0.position[0]}  end: {trace0.position[-1]}  cost: {cost0:.4f}")

    # PIBB setup
    theta_init = np.zeros(theta_dim)
    sigma_init = policy.structured_sigma(
        sigma_traj_xy=3.0,
        sigma_traj_z=0.5,
        sigma_sd=2.0,
        sigma_sk=2.0,
        sigma_ori=1.5,
    )

    # Reduce hold phase noise (phase 2: hold still)
    hold_off = policy.offsets[1]
    hold_dim = policy.theta_dims[1]
    sigma_init[hold_off:hold_off + hold_dim] *= 0.05

    optimizer = PIBB(
        theta=theta_init,
        sigma=sigma_init,
        beta=8.0,
        decay=0.99,
    )

    N_SAMPLES = 20
    N_UPDATES = 100
    best_cost  = float("inf")
    best_theta = theta_init.copy()

    print(f"\nStarting PIBB Optimization ...")
    print(f"  Task: {N_UPDATES} updates x {N_SAMPLES} samples = {N_UPDATES*N_SAMPLES} rollouts")
    for update_idx in range(N_UPDATES):
        samples = optimizer.sample(N_SAMPLES)
        costs   = np.array([
            objective_fn(policy.rollout(samples[i]))
            for i in range(N_SAMPLES)
        ])

        # Safe cost handling
        costs_safe = np.where(np.isfinite(costs), costs, 1e4)
        costs_safe = np.clip(costs_safe, 0.0, 1e4)
        optimizer.update(samples, costs_safe)

        if costs.min() < best_cost:
            best_cost  = costs.min()
            best_theta = samples[np.argmin(costs)].copy()

        if (update_idx + 1) % 10 == 0 or update_idx == 0:
            print(f"  Update {update_idx+1:03d} | "
                  f"Min: {costs.min():.4f} | "
                  f"Mean: {costs.mean():.4f} | "
                  f"BestSoFar: {best_cost:.4f}")

    print("Optimization Complete.\n")

    trace_final = policy.rollout(best_theta)
    save_checkpoint(best_theta, taskspec.horizon_sec, best_cost, trace_final,
                    checkpoint_path="scene2_pour_checkpoint.npz")
    print_diagnostics(trace_final, best_cost)

    # Generate plots
    plot_workspace(trace_final, best_cost)
    plot_trace_metrics(trace_final, best_cost)
    plot_orientation(trace_final, best_cost)
    plot_peraxis(trace_final, best_cost)


if __name__ == "__main__":
    main()
