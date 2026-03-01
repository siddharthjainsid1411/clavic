"""
Real Franka workspace — cup handover to human.

Task:
  Phase 1 (0-4 s):  Start (holding cup) -> Hold position above laptop right edge
  Phase 2 (4-6 s):  Hold still at waypoint (near-zero velocity)
  Phase 3 (6-10 s): Hold position -> Human (goal)

Plots generated:
  1. real_franka_workspace.png  — XY workspace with trajectory and obstacles
  2. real_franka_trace.png      — tr(K), tr(D), speed vs time
  3. real_franka_peraxis.png    — Per-axis Vx,Vy,Vz and Kxx,Kyy,Kzz
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from spec.compiler import Compiler
from spec.json_parser import load_taskspec_from_json
from core.multi_phase_policy import MultiPhaseCertifiedPolicy
from optimization.optimizer import PIBB
from experiment_checkpoint_warmstart import save_checkpoint
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from logic.predicates import (
    at_goal_pose, at_waypoint, hold_at_waypoint,
    obstacle_avoidance, velocity_limit,
    human_comfort_distance, human_body_exclusion,
)

# ── Scene constants ──────────────────────────────────────────────────
SAFE_Z   = 0.297
START    = np.array([0.36845, -0.07430, SAFE_Z])
HOLD     = np.array([0.61554,  0.27877, SAFE_Z])
GOAL     = np.array([0.30180,  0.72820, SAFE_Z])

LAPTOP_CORNERS_XY = np.array([
    [0.51034, 0.23381],   # #2 right-front
    [0.51542, 0.46060],   # #3 right-back
    [0.15995, 0.46442],   # #4 left-back
    [0.16797, 0.25102],   # #5 left-front
])

LAPTOP_CENTER = np.array([0.33825, 0.34875, SAFE_Z])
LAPTOP_RADIUS = 0.223

T_HOLD_START      = 4.0
T_HOLD_END        = 6.0
HOLD_WINDOW_START = T_HOLD_START + 0.05


# ── Predicate registry ───────────────────────────────────────────────
def build_predicate_registry():
    return {
        "AtGoalPose":           at_goal_pose,
        "AtWaypoint":           at_waypoint,
        "HoldAtWaypoint":       hold_at_waypoint,
        "LaptopExclusion":      obstacle_avoidance,
        "ObstacleAvoidance":    obstacle_avoidance,
        "VelocityLimit":        velocity_limit,
        "HumanComfortDistance": human_comfort_distance,
        "HumanBodyExclusion":   human_body_exclusion,
    }


# ── Diagnostics ──────────────────────────────────────────────────────
def print_diagnostics(trace, best_cost):
    pos   = trace.position
    speed = np.linalg.norm(trace.velocity, axis=1)

    mask_hold = (trace.time >= HOLD_WINDOW_START) & (trace.time <= T_HOLD_END)

    rho_goal = at_goal_pose(trace, GOAL, 0.05)
    goal_hit = trace.time[rho_goal > 0]

    rho_wp   = at_waypoint(trace, HOLD, 0.04)
    wp_hit   = trace.time[rho_wp > 0]

    lap_dist = np.linalg.norm(pos[:, :2] - LAPTOP_CENTER[:2], axis=1)
    clearance = np.min(lap_dist) - LAPTOP_RADIUS

    print("\n========== RESULT DIAGNOSTICS ==========")
    print(f"  Best cost       : {best_cost:.4f}")
    print(f"  Waypoint reached: {'YES at t=' + f'{wp_hit[0]:.2f}s' if len(wp_hit) else 'NO'}")
    print(f"  Goal reached    : {'YES at t=' + f'{goal_hit[0]:.2f}s' if len(goal_hit) else 'NO'}")
    print(f"  Max speed global: {np.max(speed):.4f} m/s  (limit 0.8 m/s)")
    print(f"  Max speed hold  : {np.max(speed[mask_hold]):.6f} m/s  (limit 0.01 m/s)")
    print(f"  Laptop clearance: {clearance*100:.1f} cm  ({'OK' if clearance >= 0 else 'VIOLATION'})")

    K_arr  = trace.gains["K"]
    D_arr  = trace.gains["D"]
    trK    = np.array([np.trace(K) for K in K_arr])
    trD    = np.array([np.trace(D) for D in D_arr])
    K_eig_min = np.min([np.linalg.eigvalsh(K)[0] for K in K_arr])
    D_eig_min = np.min([np.linalg.eigvalsh(D)[0] for D in D_arr])
    alpha = 0.05
    lyap_viol = sum(1 for D in D_arr
                    if np.any(np.linalg.eigvalsh(alpha * np.eye(3) - D) > 1e-9))
    zeta_arr  = trD / (2.0 * np.sqrt(trK))

    print(f"  tr(K) range     : [{np.min(trK):.1f}, {np.max(trK):.1f}] N/m")
    print(f"  tr(D) range     : [{np.min(trD):.3f}, {np.max(trD):.3f}] Ns/m")
    print(f"  K eig min       : {K_eig_min:.4f}  (must be > 0)")
    print(f"  D eig min       : {D_eig_min:.4f}  (must be > 0)")
    print(f"  Damping ratio ζ : [{np.min(zeta_arr):.4f}, {np.max(zeta_arr):.4f}]")
    print(f"  Lyapunov viol   : {lyap_viol}/{len(K_arr)}")
    print("=========================================\n")


# ── Plot 1: Workspace (XY view) ──────────────────────────────────────
def plot_workspace(trace, best_cost, save_path="real_franka_workspace.png"):
    pos   = trace.position
    speed = np.linalg.norm(trace.velocity, axis=1)

    fig, ax = plt.subplots(figsize=(13, 13))
    fig.suptitle(
        f"Franka Handover — Workspace (XY plane, z={SAFE_Z} m)\nbest_cost={best_cost:.3f}",
        fontsize=14, fontweight='bold'
    )

    # Laptop footprint
    lc = np.vstack([LAPTOP_CORNERS_XY, LAPTOP_CORNERS_XY[0]])
    ax.fill(LAPTOP_CORNERS_XY[:, 0], LAPTOP_CORNERS_XY[:, 1],
            color='gray', alpha=0.25, label='Laptop surface')
    ax.plot(lc[:, 0], lc[:, 1], 'k-', linewidth=1.5, alpha=0.7)

    # Exclusion circle
    circ = plt.Circle(LAPTOP_CENTER[:2], LAPTOP_RADIUS,
                      color='red', fill=True, alpha=0.1,
                      label=f'Exclusion circle (r={LAPTOP_RADIUS:.3f} m)')
    ax.add_patch(circ)
    ax.add_patch(plt.Circle(LAPTOP_CENTER[:2], LAPTOP_RADIUS,
                            color='red', fill=False, linestyle='--', linewidth=2.0))

    # Hold and goal tolerance circles
    ax.add_patch(plt.Circle(HOLD[:2], 0.04, color='blue', fill=False,
                            linestyle=':', linewidth=2.0, label='Hold tolerance (4 cm)'))
    ax.add_patch(plt.Circle(GOAL[:2], 0.05, color='black', fill=False,
                            linestyle=':', linewidth=1.5, alpha=0.6, label='Goal tolerance (5 cm)'))

    # Phase-coloured trajectory
    t_p1_end = np.searchsorted(trace.time, T_HOLD_START)
    t_p2_end = np.searchsorted(trace.time, T_HOLD_END)
    ax.plot(pos[:t_p1_end+1,  0], pos[:t_p1_end+1,  1], 'b-', lw=3, label='Phase 1: approach', zorder=3)
    ax.plot(pos[t_p1_end:t_p2_end+1, 0], pos[t_p1_end:t_p2_end+1, 1],
            'g-', lw=4, label='Phase 2: hold', zorder=4)
    ax.plot(pos[t_p2_end:, 0],  pos[t_p2_end:, 1],  'r-', lw=3, label='Phase 3: to goal', zorder=3)

    # Key points
    ax.scatter(*START[:2], s=200, c='green',  zorder=6, marker='o', edgecolor='darkgreen', linewidth=2, label='Start (cup)')
    ax.scatter(*HOLD[:2],  s=200, c='blue',   zorder=6, marker='s', edgecolor='darkblue',  linewidth=2, label='Hold waypoint')
    ax.scatter(*GOAL[:2],  s=350, c='black',  zorder=6, marker='*', label='Goal (human)')
    ax.scatter(*LAPTOP_CENTER[:2], s=100, c='red', zorder=5, marker='+', linewidth=3, label='Laptop center')

    # Corner labels
    for i, (c, lbl) in enumerate(zip(LAPTOP_CORNERS_XY, ['#2','#3','#4','#5'])):
        ax.annotate(lbl, c, fontsize=9, color='gray', fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')

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


# ── Plot 2: Trace metrics (tr(K), tr(D), speed) ───────────────────────
def plot_trace_metrics(trace, best_cost, save_path="real_franka_trace.png"):
    speed = np.linalg.norm(trace.velocity, axis=1)
    K_arr = trace.gains["K"]
    D_arr = trace.gains["D"]
    trK   = np.array([np.trace(K) for K in K_arr])
    trD   = np.array([np.trace(D) for D in D_arr])

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    fig.suptitle(
        f"Franka Handover — Impedance & Speed vs Time\nbest_cost={best_cost:.3f}",
        fontsize=12, fontweight='bold'
    )

    # ── Row 1: Speed ──────────────────────────────────────────
    ax = axes[0]
    ax.plot(trace.time, speed, 'b-', lw=2.5, label='EE speed', zorder=2)
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.15, color='green', label='Hold phase', zorder=1)
    ax.axhline(0.01, color='gray', linestyle='--', lw=1.5, alpha=0.7, label='Hold threshold (0.01 m/s)')
    ax.axhline(0.80, color='red', linestyle='--', lw=1.5, alpha=0.7, label='Speed limit (0.8 m/s)')
    for ph_t in [T_HOLD_START, T_HOLD_END]:
        ax.axvline(ph_t, color='orange', linestyle=':', lw=1.2, alpha=0.6)
    ax.set_ylabel("Speed (m/s)", fontsize=10, fontweight='bold')
    ax.set_title("End-Effector Speed", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 10)

    # ── Row 2: Stiffness tr(K) ────────────────────────────────
    ax = axes[1]
    ax.plot(trace.time, trK, 'm-', lw=2.5, label='tr(K) = sum of diagonals', zorder=2)
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.15, color='green', label='Hold phase', zorder=1)
    for ph_t in [T_HOLD_START, T_HOLD_END]:
        ax.axvline(ph_t, color='orange', linestyle=':', lw=1.2, alpha=0.6)
    ax.axhline(200, color='gray', linestyle='--', lw=1, alpha=0.5, label='Initial K0=200 N/m')
    ax.set_ylabel("tr(K) (N/m)", fontsize=10, fontweight='bold')
    ax.set_title("Stiffness Evolution", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 10)

    # ── Row 3: Damping tr(D) ──────────────────────────────────
    ax = axes[2]
    ax.plot(trace.time, trD, 'c-', lw=2.5, label='tr(D) = sum of diagonals', zorder=2)
    ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.15, color='green', label='Hold phase', zorder=1)
    ax.axhline(30.0, color='orange', linestyle='--', lw=1.5, alpha=0.7, label='Min safe tr(D)=30 Ns/m')
    for ph_t in [T_HOLD_START, T_HOLD_END]:
        ax.axvline(ph_t, color='orange', linestyle=':', lw=1.2, alpha=0.6)
    ax.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
    ax.set_ylabel("tr(D) (Ns/m)", fontsize=10, fontweight='bold')
    ax.set_title("Damping Evolution", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved -> {save_path}")
    plt.close()


# ── Plot 3: Per-axis velocities and stiffnesses ──────────────────────
def plot_peraxis(trace, best_cost, save_path="real_franka_peraxis.png"):
    vel    = trace.velocity
    K_arr  = trace.gains["K"]
    K_diag = np.array([np.diag(K) for K in K_arr])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Franka Handover — Per-Axis Velocities & Stiffnesses\nbest_cost={best_cost:.3f}",
        fontsize=12, fontweight='bold'
    )
    axis_labels = ['X', 'Y', 'Z']
    vel_colors  = ['steelblue', 'darkorange', 'seagreen']
    k_colors    = ['mediumpurple', 'tomato', 'goldenrod']

    for i, (lbl, vc, kc) in enumerate(zip(axis_labels, vel_colors, k_colors)):
        # Row 0: velocity components
        ax = axes[0, i]
        ax.plot(trace.time, vel[:, i], color=vc, lw=2.5, label=f'V{lbl}', zorder=2)
        ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.15, color='green', label='Hold phase', zorder=1)
        ax.axhline(0.0, color='gray', linestyle='--', lw=0.8, alpha=0.5)
        for ph_t in [T_HOLD_START, T_HOLD_END]:
            ax.axvline(ph_t, color='orange', linestyle=':', lw=1.0, alpha=0.6)
        ax.set_ylabel(f"V{lbl} (m/s)", fontsize=10, fontweight='bold')
        ax.set_title(f"Velocity {lbl}-axis", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 10)

        # Row 1: stiffness diagonal
        ax = axes[1, i]
        ax.plot(trace.time, K_diag[:, i], color=kc, lw=2.5, label=f'K{lbl}{lbl}', zorder=2)
        ax.axvspan(T_HOLD_START, T_HOLD_END, alpha=0.15, color='green', label='Hold phase', zorder=1)
        ax.axhline(200.0, color='gray', linestyle='--', lw=1, alpha=0.5, label='K0=200 N/m')
        for ph_t in [T_HOLD_START, T_HOLD_END]:
            ax.axvline(ph_t, color='orange', linestyle=':', lw=1.0, alpha=0.6)
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


# ── CSV Export (1kHz trajectory) ──────────────────────────────────────
def export_trajectory_csv(trace, save_path="franka_trajectory.csv"):
    """
    Export trajectory to CSV at 1kHz.
    
    Format (24 columns):
      x,y,z,dx,dy,dz, [9 K elements], [9 D elements]
    
    Vx, Vy, Vz are included so they can be used if needed on hardware.
    """
    pos   = trace.position
    vel   = trace.velocity
    K_arr = trace.gains["K"]
    D_arr = trace.gains["D"]
    t_100 = trace.time
    
    # Flatten 3×3 matrices
    K_flat = K_arr.reshape(len(K_arr), 9)
    D_flat = D_arr.reshape(len(D_arr), 9)
    
    # Gaussian smooth (σ=3 at 100Hz = 30ms window)
    K_smooth = np.array([gaussian_filter1d(K_flat[:, i], sigma=3) for i in range(9)]).T
    D_smooth = np.array([gaussian_filter1d(D_flat[:, i], sigma=3) for i in range(9)]).T
    
    # Re-symmetrize to preserve 3×3 structure
    K_smooth_3d = K_smooth.reshape(len(K_smooth), 3, 3)
    for i in range(len(K_smooth_3d)):
        K_smooth_3d[i] = (K_smooth_3d[i] + K_smooth_3d[i].T) / 2.0
    K_smooth = K_smooth_3d.reshape(len(K_smooth_3d), 9)
    
    D_smooth_3d = D_smooth.reshape(len(D_smooth), 3, 3)
    for i in range(len(D_smooth_3d)):
        D_smooth_3d[i] = (D_smooth_3d[i] + D_smooth_3d[i].T) / 2.0
    D_smooth = D_smooth_3d.reshape(len(D_smooth_3d), 9)
    
    # Cubic spline upsample to 1kHz (10001 steps)
    t_new = np.linspace(0, 10.0, 10001)
    cs_pos = CubicSpline(t_100, pos)
    cs_vel = CubicSpline(t_100, vel)
    cs_K = CubicSpline(t_100, K_smooth)
    cs_D = CubicSpline(t_100, D_smooth)
    
    pos_1k = cs_pos(t_new)
    vel_1k = cs_vel(t_new)
    K_1k   = cs_K(t_new)
    D_1k   = cs_D(t_new)
    
    # Assemble 24-column array
    rows = np.hstack([pos_1k, vel_1k, K_1k, D_1k])
    
    # Header with column names
    header = 'x,y,z,dx,dy,dz,' + ','.join(f'k{i//3+1}{i%3+1}' for i in range(9)) + ',' + \
             ','.join(f'd{i//3+1}{i%3+1}' for i in range(9))
    
    np.savetxt(save_path, rows, delimiter=',', header=header, comments='', fmt='%.8f')
    print(f"✓ CSV exported: {save_path} ({rows.shape[0]} rows × {rows.shape[1]} columns)")
    
    # Verify hardware safety
    print("\n  === CSV Hardware Safety Check ===")
    accel = np.linalg.norm(np.gradient(vel_1k, t_new, axis=0), axis=1)
    K_pos_def = all(np.linalg.eigvalsh(K_1k[i].reshape(3, 3))[0] > 0 for i in range(0, len(K_1k), 100))
    dKdt_max = np.max(np.abs(np.gradient(K_1k, t_new, axis=0)))
    
    print(f"  Max accel:      {np.max(accel):.3f} m/s²  (limit 13.0) → {'OK' if np.max(accel) < 13.0 else 'FAIL'}")
    print(f"  K always PD:    {K_pos_def} → {'OK' if K_pos_def else 'FAIL'}")
    print(f"  Max dK/dt:      {dKdt_max:.1f} N/m/s  (limit ~5000) → {'OK' if dKdt_max < 5000 else 'WARN'}")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    taskspec = load_taskspec_from_json("spec/real_franka_task.json")
    assert taskspec.phases is not None, "real_franka_task.json must contain 'phases'"
    
    policy    = MultiPhaseCertifiedPolicy(taskspec.phases)
    theta_dim = policy.parameter_dimension()
    print(f"Multi-phase policy: {len(taskspec.phases)} phases, theta_dim = {theta_dim}")

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
    )

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
    N_UPDATES = 100       # Changed from 200 to 100
    best_cost  = float("inf")
    best_theta = theta_init.copy()

    print("\nStarting PIBB Optimization ...")
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
    save_checkpoint(best_theta, taskspec.horizon_sec, best_cost, trace_final)
    print_diagnostics(trace_final, best_cost)
    
    # Generate 3 separate plots
    plot_workspace(trace_final, best_cost, save_path="real_franka_workspace.png")
    plot_trace_metrics(trace_final, best_cost, save_path="real_franka_trace.png")
    plot_peraxis(trace_final, best_cost, save_path="real_franka_peraxis.png")
    
    # Export trajectory to CSV (automatic, 1kHz, with velocities)
    export_trajectory_csv(trace_final, save_path="franka_trajectory.csv")


if __name__ == "__main__":
    main()
