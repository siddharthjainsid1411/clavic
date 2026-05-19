"""
Exp 2: Carry → Hold-at-Waypoint (2 s) → Continue to Goal.

3-phase task:
  Phase 1 (0–5 s) : carry from start, avoid obstacle (HARD), reach waypoint
  Phase 2 (5–7 s) : hold at intermediate waypoint — velocity goes to zero
  Phase 3 (7–11 s): resume motion, reach final goal

Key features vs Scene 3:
  - THREE phases: carry | hold-2s | continue  (Scene 3 had carry | pour)
  - Obstacle avoidance = HARD throughout
      → avoidance="HARD"  in set_obstacles()
      → DMP repulsive forcing (steers ODE organically around obstacle)
      → hard radial projector post-rollout (backstop guarantee)
      → GUARANTEED  ||p(t) − c|| ≥ r_safe  ∀t
    - Constant orientation from spec/exp2_task.json throughout all 3 phases (no tilt, no pour)
  - No human proximity stiffness penalty — goal is a delivery point, not a person
    - Geometry is loaded from spec/exp2_task.json (start, waypoint, goal, obstacle)

Plots (PNG only, 300 dpi):
  1. scene4_workspace.png   — 3D Franka FRS view
  2. scene4_topdown.png     — 2D X–Y top-down view
  3. scene4_stiffness.png   — Per-axis Kxx/Kyy/Kzz vs time
  4. scene4_orientation.png — Euler angles vs time (all near 0)
    5. scene4_orientation_geodesic.png — angle to q_ref vs time
    6. scene4_kinematics.png  — Position & velocity time-series
    7. scene4_angular_velocity.png — omega norm vs time
    8. scene4_velocity_cbf.png — speed vs time-windowed vmax
    9. scene4_acceleration.png — linear & angular acceleration
    10. scene4_obstacle_hocbf.png — obstacle HOCBF h, hdot, and active masks
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import seaborn as sns

from spec.compiler import Compiler
from spec.json_parser import load_taskspec_from_json
from core.multi_phase_policy import MultiPhaseCertifiedPolicy
from optimization.optimizer import PIBB
from core.cgms.quat_utils import quat_normalize

from logic.predicates import (
    at_waypoint, hold_at_waypoint,
    obstacle_avoidance, velocity_limit,
    orientation_limit, angular_velocity_limit,
)

# ── style ──────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# ── scene constants (loaded from JSON in main()) ──────────────────────
START    = None
WAYPOINT = None
GOAL     = None
OBSTACLE = None
OBS_RAD      = 0.12
OBS_SAFE_RAD = 0.12
OBSTACLE_GEOMETRY = "sphere"
VELOCITY_LIMIT = 0.8

Q_UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])
ORI_REF = np.array([1.0, 0.0, 0.0, 0.0])
ORI_MAX_ANGLE = 0.15

# Phase timing (loaded from JSON in main())
T_CARRY_END = 5.0
T_HOLD_END  = 7.0
T_CONT_END  = 11.0

HUMAN_POS      = np.zeros(3)
HUMAN_PROX_RAD = 0.12
HUMAN_RAMP_RAD = 0.36
K_AXIS_LIMIT   = 100.0

OBS_HX, OBS_HY, OBS_HZ = 0.08, 0.08, 0.05

# ── colours ────────────────────────────────────────────────────────────
C_CARRY  = "#4C72B0"   # blue  — carry phase
C_HOLD   = "#DD8452"   # orange — hold phase
C_CONT   = "#55A868"   # green  — continue phase
C_OBS    = "#AAAAAA"
C_HUMAN  = "#AEC7E8"
C_START  = "#2CA02C"
C_WP     = "#9467BD"   # purple — waypoint marker
C_GOAL   = "#1F77B4"
C_DASH   = "#999999"
C_KX     = "#4C72B0"
C_KY     = "#DD8452"
C_KZ     = "#55A868"

SCENE_LABEL = "Scene 4 — Carry → Hold (2 s) → Continue (HARD obstacle avoidance)"


# ── predicate registry ────────────────────────────────────────────────
def build_predicate_registry():
    return {
        "AtWaypoint":           at_waypoint,
        "HoldAtWaypoint":       hold_at_waypoint,
        "AtGoal":               at_waypoint,        # same fn, different binding
        "HoldAtGoal":           hold_at_waypoint,   # same fn, different binding
        "ObstacleAvoidance":    obstacle_avoidance,
        "VelocityLimit":        velocity_limit,
        "OrientationLimit":     orientation_limit,
        "AngularVelocityLimit": angular_velocity_limit,
    }


# ── quaternion → Euler ZYX (degrees) ─────────────────────────────────
def quat_to_euler(q):
    w, x, y, z = q
    roll  = np.degrees(np.arctan2(2*(w*x + y*z),  1 - 2*(x*x + y*y)))
    sinp  = np.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y),  1 - 2*(y*y + z*z)))
    return roll, pitch, yaw


def quat_geodesic_angle(q, q_ref):
    """Return the geodesic angle between two quaternions (radians)."""
    q = quat_normalize(np.asarray(q, float))
    q_ref = quat_normalize(np.asarray(q_ref, float))
    dot = float(np.clip(abs(np.dot(q, q_ref)), -1.0, 1.0))
    return 2.0 * np.arccos(dot)


# ── diagnostics ───────────────────────────────────────────────────────
def print_diagnostics(trace, best_cost):
    pos   = trace.position
    speed = np.linalg.norm(trace.velocity, axis=1)
    K_arr = trace.gains["K"]
    D_arr = trace.gains["D"]
    t     = trace.time

    trK   = np.array([np.trace(K) for K in K_arr])
    trD   = np.array([np.trace(D) for D in D_arr])

    K_diag = np.array([[K[0,0], K[1,1], K[2,2]] for K in K_arr])
    K_eig_min = min(np.linalg.eigvalsh(K)[0] for K in K_arr)

    if OBSTACLE_GEOMETRY == "cylinder_infinite":
        d_obs = np.linalg.norm(pos[:, :2] - OBSTACLE[:2], axis=1)
    else:
        d_obs = np.linalg.norm(pos - OBSTACLE, axis=1)
    obs_cm = (d_obs.min() - OBS_RAD) * 100.0
    n_inside = int(np.sum(d_obs < OBS_RAD))

    # waypoint reach
    d_wp   = np.linalg.norm(pos - WAYPOINT, axis=1)
    wp_reached = np.any(d_wp < 0.06)
    t_wp = float(t[np.argmin(d_wp)]) if wp_reached else float("nan")

    # goal reach
    d_goal = np.linalg.norm(pos - GOAL, axis=1)
    reached = np.any(d_goal < 0.06)
    t_reach = float(t[np.argmin(d_goal)]) if reached else float("nan")

    # hold phase velocity
    hold_mask = (t >= T_CARRY_END) & (t <= T_HOLD_END)
    hold_speed_max = float(speed[hold_mask].max()) if np.any(hold_mask) else float("nan")

    # hold position drift at waypoint during hold phase
    hold_drift = float(d_wp[hold_mask].max()) if np.any(hold_mask) else float("nan")
    velocity_cbf = trace.safety.get("velocity_cbf", {}) if hasattr(trace, "safety") else {}
    cbf_active = int(np.sum(velocity_cbf.get("active", []))) if velocity_cbf else 0
    cbf_viol = int(np.sum(velocity_cbf.get("post_projection_violation", []))) if velocity_cbf else 0
    cbf_min_h = None
    if velocity_cbf and "post_projection_h" in velocity_cbf:
        cbf_min_h = float(np.nanmin(velocity_cbf["post_projection_h"]))
    window_active = int(np.sum(velocity_cbf.get("window_active", []))) if velocity_cbf else 0
    obstacle_hocbf = trace.safety.get("obstacle_hocbf", {}) if hasattr(trace, "safety") else {}
    obs_h_min = None
    obs_hdot_min = None
    obs_active_steps = 0
    if obstacle_hocbf:
        if "h" in obstacle_hocbf:
            h_vals = np.asarray(obstacle_hocbf["h"], dtype=float)
            if np.any(np.isfinite(h_vals)):
                obs_h_min = float(np.nanmin(h_vals))
        if "hdot" in obstacle_hocbf:
            hdot_vals = np.asarray(obstacle_hocbf["hdot"], dtype=float)
            if np.any(np.isfinite(hdot_vals)):
                obs_hdot_min = float(np.nanmin(hdot_vals))
        obs_active_steps = int(np.sum(obstacle_hocbf.get("active", [])))
    proj = trace.safety.get("obstacle_projection", {}) if hasattr(trace, "safety") else {}
    proj_active_steps = int(np.sum(proj.get("active", []))) if proj else 0

    def _mask_intervals(times, mask, max_show=6):
        if mask is None or len(mask) == 0:
            return "none"
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return "none"
        idx = np.where(mask)[0]
        runs = []
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
                continue
            runs.append((times[start], times[prev]))
            start = i
            prev = i
        runs.append((times[start], times[prev]))
        parts = [f"[{s:.2f},{e:.2f}]" for s, e in runs[:max_show]]
        if len(runs) > max_show:
            parts.append("...")
        return ", ".join(parts)

    # orientation geodesic angle diagnostics
    ori_angle_max = float("nan")
    ori_viol = 0
    if trace.orientation is not None:
        angles = np.array([quat_geodesic_angle(q, ORI_REF) for q in trace.orientation])
        ori_angle_max = float(np.max(angles))
        ori_viol = int(np.sum(angles > ORI_MAX_ANGLE + 1e-9))

    # angular velocity diagnostics
    omega_max = float("nan")
    omega_viol = 0
    if trace.angular_velocity is not None:
        omega_norm = np.linalg.norm(trace.angular_velocity, axis=1)
        omega_max = float(np.max(omega_norm))
        av = trace.safety.get("angular_velocity_cbf", {}) if hasattr(trace, "safety") else {}
        if av and "post_integration_violation" in av:
            omega_viol = int(np.sum(av["post_integration_violation"]))

    sep = "=" * 48
    print(f"\n{sep} EXP 2 DIAGNOSTICS {sep}")
    print(f"  Scene             : Carry → Hold-2s → Continue (HARD obstacle)")
    print(f"  Best cost         : {best_cost:.4f}")
    print(f"  Waypoint reached  : {'YES' if wp_reached else 'NO'}  t={t_wp:.2f} s")
    print(f"  Hold speed (max)  : {hold_speed_max:.4f} m/s  (target ≈ 0)")
    print(f"  Hold drift (max)  : {hold_drift*100:.1f} cm from waypoint during hold")
    print(f"  Goal reached      : {'YES' if reached else 'NO'}  t={t_reach:.2f} s")
    print(f"  Max speed         : {speed.max():.4f} m/s  (limit {VELOCITY_LIMIT:.3f})")
    print(f"  Velocity CBF      : active_steps={cbf_active}, window_on={window_active}, violations={cbf_viol}, min_h={cbf_min_h}")
    if velocity_cbf:
        print(f"  CBF active windows: {_mask_intervals(t, velocity_cbf.get('active', []))}")
        print(f"  Limit windows     : {_mask_intervals(t, velocity_cbf.get('window_active', []))}")
    print(f"  Orientation angle : max={ori_angle_max:.4f} rad  (limit {ORI_MAX_ANGLE:.3f}), violations={ori_viol}")
    print(f"  Angular velocity  : max={omega_max:.4f} rad/s, violations={omega_viol}")
    oh = trace.safety.get("orientation_hocbf", {}) if hasattr(trace, "safety") else {}
    if oh:
        print(f"  Ori HOCBF active  : {_mask_intervals(t, oh.get('active', []))}")
    av = trace.safety.get("angular_velocity_cbf", {}) if hasattr(trace, "safety") else {}
    if av:
        print(f"  AngVel CBF active : {_mask_intervals(t, av.get('active', []))}")
    if obstacle_hocbf:
        print(
            f"  Obstacle HOCBF    : active_steps={obs_active_steps}, "
            f"min_h={obs_h_min}, min_hdot={obs_hdot_min}"
        )
        print(f"  Obstacle HOCBF on : {_mask_intervals(t, obstacle_hocbf.get('active', []))}")
    if proj:
        print(f"  Projection active : steps={proj_active_steps}")
        print(f"  Projection windows: {_mask_intervals(t, proj.get('active', []))}")
    print(f"  Obstacle clearance: {obs_cm:.1f} cm  (HARD {OBSTACLE_GEOMETRY}, r={OBS_RAD:.2f} m)")
    print(f"  Pts inside obs    : {n_inside}  (HARD: must be 0)")
    print(f"  tr(K) range       : [{trK.min():.0f}, {trK.max():.0f}] N/m")
    print(f"  tr(D) range       : [{trD.min():.1f}, {trD.max():.1f}] Ns/m")
    print(f"  K eigenvalue min  : {K_eig_min:.4f}  (CGMS > 0 required)")
    print(f"  Note              : No human proximity penalty — goal is delivery point only")
    print("=" * (48*2 + len(" EXP 2 DIAGNOSTICS ")))


# ======================================================================
#  PLOT helpers
# ======================================================================
def _phase_indices(t):
    """Return (i_carry_end, i_hold_end) index into t array."""
    return np.searchsorted(t, T_CARRY_END), np.searchsorted(t, T_HOLD_END)


def _obs_box_verts(cx, cy, cz, dx, dy, dz):
    corners = np.array([
        [cx-dx, cy-dy, cz-dz], [cx+dx, cy-dy, cz-dz],
        [cx+dx, cy+dy, cz-dz], [cx-dx, cy+dy, cz-dz],
        [cx-dx, cy-dy, cz+dz], [cx+dx, cy-dy, cz+dz],
        [cx+dx, cy+dy, cz+dz], [cx-dx, cy+dy, cz+dz],
    ])
    return corners


# ======================================================================
#  PLOT 1 — 3D workspace
# ======================================================================
def plot_3d_workspace(trace, best_cost, base="exp2_workspace"):
    pos = trace.position
    t   = trace.time

    # coordinate remap: plot axes = (Y_world, X_world, Z_world)
    def to_plot(p):
        p = np.asarray(p)
        if p.ndim == 1:
            return np.array([p[1], p[0], p[2]])
        return p[:, [1, 0, 2]]

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # obstacle box: bottom 20% solid black (obstacle base), top 80% translucent
    cx, cy, cz = to_plot(OBSTACLE)
    dx, dy, dz = OBS_HY, OBS_HX, OBS_HZ
    split_z = -dz + 2 * dz * 0.20   # z-offset at 20% of total height

    def _split_faces(z_lo, z_hi):
        v = _obs_box_verts(cx, cy, cz, dx, dy, 0.0)   # get XY corners at cz
        # rebuild with explicit z range
        c = np.array([
            [cx-dx, cy-dy, cz+z_lo], [cx+dx, cy-dy, cz+z_lo],
            [cx+dx, cy+dy, cz+z_lo], [cx-dx, cy+dy, cz+z_lo],
            [cx-dx, cy-dy, cz+z_hi], [cx+dx, cy-dy, cz+z_hi],
            [cx+dx, cy+dy, cz+z_hi], [cx-dx, cy+dy, cz+z_hi],
        ])
        return [
            [c[0],c[1],c[5],c[4]], [c[2],c[3],c[7],c[6]],
            [c[0],c[3],c[7],c[4]], [c[1],c[2],c[6],c[5]],
            [c[4],c[5],c[6],c[7]], [c[0],c[1],c[2],c[3]],
        ]

    ax.add_collection3d(Poly3DCollection(
        _split_faces(-dz, split_z),
        alpha=0.90, facecolor="#1A1A1A", edgecolor="#000000", linewidth=0.7
    ))
    ax.add_collection3d(Poly3DCollection(
        _split_faces(split_z, +dz),
        alpha=0.22, facecolor=C_OBS, edgecolor="#666666", linewidth=0.5
    ))

    gp = to_plot(GOAL)

    # straight start→goal path
    sp = to_plot(START)
    ax.plot([sp[0], gp[0]], [sp[1], gp[1]], [sp[2], gp[2]],
            "--", color=C_DASH, lw=1.6, alpha=0.55, label="Shortest path", zorder=2)

    # trajectory coloured by phase
    pp = to_plot(pos)
    i1, i2 = _phase_indices(t)
    ax.plot(pp[:i1+1, 0], pp[:i1+1, 1], pp[:i1+1, 2],
            color=C_CARRY, lw=2.2, solid_capstyle="round", zorder=5, label="Carry")
    ax.plot(pp[i1:i2+1, 0], pp[i1:i2+1, 1], pp[i1:i2+1, 2],
            color=C_HOLD,  lw=2.8, solid_capstyle="round", zorder=6, label="Hold (2 s)")
    ax.plot(pp[i2:, 0], pp[i2:, 1], pp[i2:, 2],
            color=C_CONT,  lw=2.2, solid_capstyle="round", zorder=5, label="Continue")

    # markers
    ax.scatter(*sp, s=65, c=C_START, zorder=10, depthshade=False,
               edgecolors="black", linewidth=0.6)
    ax.text(sp[0]+0.02, sp[1]-0.01, sp[2]+0.03, "Start",
            fontsize=8, fontweight="bold", color=C_START)

    wp_p = to_plot(WAYPOINT)
    ax.scatter(*wp_p, s=80, c=C_WP, zorder=10, depthshade=False,
               marker="^", edgecolors="black", linewidth=0.6)
    ax.text(wp_p[0]+0.02, wp_p[1]+0.01, wp_p[2]+0.03, "Waypoint\n(hold 2s)",
            fontsize=8, fontweight="bold", color=C_WP)

    ax.scatter(*gp, s=60, c=C_GOAL, zorder=10, depthshade=False,
               marker="D", edgecolors="black", linewidth=0.6)
    ax.text(gp[0]+0.03, gp[1]+0.01, gp[2]+0.03, "Goal",
            fontsize=8, fontweight="bold", color=C_GOAL)

    obs_p = to_plot(OBSTACLE)
    ax.text(obs_p[0], obs_p[1]+0.04, obs_p[2]+0.03, "Obstacle\n(HARD)",
            fontsize=7, color="#333333", ha="center", fontweight="bold")

    ax.set_xlabel("Y — lateral (m)",      fontsize=9, labelpad=7)
    ax.set_ylabel("X — forward (m)",      fontsize=9, labelpad=7)
    ax.set_zlabel("Z — height (m)",        fontsize=9, labelpad=7)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=22, azim=-50)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("lightgrey")
    ax.grid(True, alpha=0.28)

    extra = [
        mpatches.Patch(facecolor="#1A1A1A", alpha=0.90, edgecolor="#000",
                       label="Obstacle base (solid)"),
        mpatches.Patch(facecolor=C_OBS,     alpha=0.30, edgecolor="#666",
                       label="Obstacle upper (HARD avoidance)"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, 0.97), framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)
    ax.set_title(SCENE_LABEL, fontsize=9, pad=10)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ======================================================================
#  PLOT 2 — 2D top-down view
# ======================================================================
def plot_2d_topdown(trace, best_cost, base="exp2_topdown"):
    pos = trace.position
    t   = trace.time

    fig, ax = plt.subplots(figsize=(7, 6.5))

    side_half = max(OBS_HX, OBS_HY)
    obs_square = mpatches.Rectangle(
        (OBSTACLE[0] - side_half, OBSTACLE[1] - side_half),
        2 * side_half,
        2 * side_half,
        facecolor=C_OBS,
        edgecolor="#666666",
        linewidth=1.0,
        alpha=0.30,
        zorder=2,
        label="Obstacle footprint (square)",
    )
    obs_edge = plt.Circle((OBSTACLE[0], OBSTACLE[1]), OBS_SAFE_RAD,
                          color="#333333", alpha=0.0, fill=False,
                          linestyle="-", linewidth=1.4, zorder=3,
                          label=f"HARD model circle (r={OBS_SAFE_RAD:.2f} m)")
    ax.add_patch(obs_square)
    ax.add_patch(obs_edge)
    ax.text(OBSTACLE[0], OBSTACLE[1] + side_half + 0.025,
            "Obstacle\n(square plot, circle guarantee)",
            fontsize=7.5, ha="center", color="#333333", fontweight="bold")

    # straight start→goal
    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]],
            "--", color=C_DASH, lw=1.5, alpha=0.50, zorder=3, label="Shortest path")

    # trajectory by phase
    i1, i2 = _phase_indices(t)
    ax.plot(pos[:i1+1, 0], pos[:i1+1, 1],
            color=C_CARRY, lw=2.2, solid_capstyle="round", zorder=5, label="Carry")
    ax.plot(pos[i1:i2+1, 0], pos[i1:i2+1, 1],
            color=C_HOLD, lw=3.0, solid_capstyle="round", zorder=6, label="Hold (2 s)")
    ax.plot(pos[i2:, 0], pos[i2:, 1],
            color=C_CONT, lw=2.2, solid_capstyle="round", zorder=5, label="Continue")

    # markers
    ax.scatter(START[0],    START[1],    s=80, c=C_START, zorder=10,
               edgecolors="black", linewidth=0.7, label="Start")
    ax.scatter(WAYPOINT[0], WAYPOINT[1], s=90, c=C_WP, zorder=10,
               marker="^", edgecolors="black", linewidth=0.7, label="Waypoint (hold)")
    ax.scatter(GOAL[0],     GOAL[1],     s=75, c=C_GOAL, zorder=10,
               marker="D", edgecolors="black", linewidth=0.7, label="Goal")

    ax.text(START[0]+0.01,    START[1]-0.04,    "Start",    fontsize=8.5,
            fontweight="bold", color=C_START)
    ax.text(WAYPOINT[0]-0.08, WAYPOINT[1]+0.01, "Waypoint\n(hold 2s)", fontsize=8,
            fontweight="bold", color=C_WP)
    ax.text(GOAL[0]+0.01,     GOAL[1]+0.01,     "Goal", fontsize=8,
            fontweight="bold", color=C_GOAL)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(f"Top-down view (X–Y)\n{SCENE_LABEL}", fontsize=10)
    ax.set_aspect("equal")

    margin = 0.10
    ax.set_xlim(min(START[0], GOAL[0], WAYPOINT[0]) - margin,
                max(START[0], GOAL[0], WAYPOINT[0]) + margin + 0.05)
    ax.set_ylim(min(START[1], GOAL[1], WAYPOINT[1]) - margin,
                max(START[1], GOAL[1], WAYPOINT[1]) + margin + 0.12)

    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ======================================================================
#  PLOT 3 — per-axis stiffness vs time
# ======================================================================
def plot_stiffness(trace, best_cost, base="exp2_stiffness"):
    K_arr = trace.gains["K"]
    Kd    = np.array([[K[0,0], K[1,1], K[2,2]] for K in K_arr])
    t     = trace.time

    fig, ax = plt.subplots(figsize=(10, 4.2))

    # phase backgrounds (no human proximity bands — goal is not a human)
    ax.axvspan(0,           T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
    ax.axvspan(T_CARRY_END, T_HOLD_END,  alpha=0.040, color=C_HOLD,  zorder=0)
    ax.axvspan(T_HOLD_END,  T_CONT_END,  alpha=0.025, color=C_CONT,  zorder=0)

    # hold phase highlight
    ax.axvspan(T_CARRY_END, T_HOLD_END, alpha=0.12, color=C_HOLD, zorder=0,
               label="Hold phase (velocity → 0)")

    ax.plot(t, Kd[:,0], color=C_KX, lw=2.0, label=r"$K_{xx}$")
    ax.plot(t, Kd[:,1], color=C_KY, lw=2.0, label=r"$K_{yy}$")
    ax.plot(t, Kd[:,2], color=C_KZ, lw=2.0, label=r"$K_{zz}$")

    yhi = Kd.max() * 1.15 + 20.0
    ax.set_ylim(0, yhi)
    for tx, lb, col in [(2.5, "Carry", C_CARRY),
                        (6.0, "Hold", C_HOLD),
                        (9.0, "Continue", C_CONT)]:
        ax.text(tx, yhi * 0.94, lb, fontsize=8, color=col,
                ha="center", fontweight="bold", alpha=0.75)

    # phase boundaries
    for tv in [T_CARRY_END, T_HOLD_END]:
        ax.axvline(tv, color="#888888", lw=0.8, ls="--", alpha=0.50)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Stiffness (N/m)", fontsize=11)
    ax.set_title(SCENE_LABEL, fontsize=9)
    ax.set_xlim(0, T_CONT_END)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ======================================================================
#  PLOT 4 — Euler orientation vs time (should stay near 0 throughout)
# ======================================================================
def plot_orientation_euler(trace, best_cost, base="exp2_orientation"):
    if trace.orientation is None:
        print("No orientation — skipping.")
        return
    q     = trace.orientation
    t     = trace.time
    euler = np.array([quat_to_euler(q[k]) for k in range(len(t))])
    roll, pitch, yaw = euler[:,0], euler[:,1], euler[:,2]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(0,            T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
    ax.axvspan(T_CARRY_END,  T_HOLD_END,  alpha=0.040, color=C_HOLD,  zorder=0)
    ax.axvspan(T_HOLD_END,   T_CONT_END,  alpha=0.025, color=C_CONT,  zorder=0)
    ax.axvspan(T_CARRY_END,  T_HOLD_END,  alpha=0.10,  color=C_HOLD,  zorder=0)

    ax.plot(t, roll,  color=C_KX, lw=1.4, ls="--", alpha=0.60, label=r"Roll $\theta_x$")
    ax.plot(t, yaw,   color=C_KZ, lw=1.4, ls="--", alpha=0.60, label=r"Yaw $\theta_z$")
    ax.plot(t, pitch, color=C_KY, lw=2.2, label=r"Pitch $\theta_y$")
    ax.axhline(0.0, color="#999999", ls="-", lw=0.5, alpha=0.30)

    # Note: Euler angles are absolute and can wrap by ±180°.
    # The true constraint is enforced on the geodesic angle to q_ref.

    all_max = max(abs(roll).max(), abs(pitch).max(), abs(yaw).max(), 10.0)
    ax.set_ylim(-all_max * 1.2, all_max * 1.2)
    yhi = ax.get_ylim()[1]

    for tv in [T_CARRY_END, T_HOLD_END]:
        ax.axvline(tv, color="#888888", lw=0.8, ls="--", alpha=0.50)
    for tx, lb, col in [(2.5, "Carry", C_CARRY),
                        (6.0, "Hold",  C_HOLD),
                        (9.0, "Continue", C_CONT)]:
        ax.text(tx, yhi * 0.88, lb, fontsize=8, color=col,
                ha="center", fontweight="bold", alpha=0.75)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Angle (deg)", fontsize=11)
    qtxt = f"[{ORI_REF[0]:.3f},{ORI_REF[1]:.3f},{ORI_REF[2]:.3f},{ORI_REF[3]:.3f}]"
    ax.set_title(f"{SCENE_LABEL}\nEuler angles (absolute); reference q_ref={qtxt}", fontsize=9)
    ax.set_xlim(0, T_CONT_END)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# =====================================================================
#  PLOT 4b — Geodesic angle to q_ref vs time (true constraint)
# =====================================================================
def plot_orientation_geodesic(trace, best_cost, base="exp2_orientation_geodesic"):
    if trace.orientation is None:
        print("No orientation — skipping geodesic plot.")
        return
    t = trace.time
    angles = np.array([quat_geodesic_angle(q, ORI_REF) for q in trace.orientation])
    angles_deg = np.degrees(angles)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(0,            T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
    ax.axvspan(T_CARRY_END,  T_HOLD_END,  alpha=0.040, color=C_HOLD,  zorder=0)
    ax.axvspan(T_HOLD_END,   T_CONT_END,  alpha=0.025, color=C_CONT,  zorder=0)

    ax.plot(t, angles_deg, color=C_KY, lw=2.2, label="Geodesic angle to q_ref")

    lim_deg = np.degrees(ORI_MAX_ANGLE)
    ax.axhline(lim_deg, color="#CC4444", ls=":", lw=1.0, alpha=0.70,
               label=f"limit = {lim_deg:.1f}°")

    for tv in [T_CARRY_END, T_HOLD_END]:
        ax.axvline(tv, color="#888888", lw=0.8, ls="--", alpha=0.50)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Angle to q_ref (deg)", fontsize=11)
    ax.set_title(f"{SCENE_LABEL}\nGeodesic angle to q_ref", fontsize=9)
    ax.set_xlim(0, T_CONT_END)
    ax.set_ylim(0.0, max(angles_deg.max() * 1.15, lim_deg * 1.3, 5.0))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ======================================================================
#  PLOT 5 — position & velocity time-series
# ======================================================================
def plot_kinematics(trace, best_cost, base="exp2_kinematics"):
    pos   = trace.position
    vel   = trace.velocity
    t     = trace.time
    speed = np.linalg.norm(vel, axis=1)
    if OBSTACLE_GEOMETRY == "cylinder_infinite":
        d_obs = np.linalg.norm(pos[:, :2] - OBSTACLE[:2], axis=1)
    else:
        d_obs = np.linalg.norm(pos - OBSTACLE, axis=1)
    d_wp  = np.linalg.norm(pos - WAYPOINT, axis=1)
    d_goal= np.linalg.norm(pos - GOAL,     axis=1)

    c_x = "#4C72B0"; c_y = "#DD8452"; c_z = "#55A868"; c_spd = "#8172B2"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for axi in axes:
        axi.axvspan(0,            T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
        axi.axvspan(T_CARRY_END,  T_HOLD_END,  alpha=0.060, color=C_HOLD,  zorder=0)
        axi.axvspan(T_HOLD_END,   T_CONT_END,  alpha=0.025, color=C_CONT,  zorder=0)

        # obstacle proximity band
        obs_near = d_obs < (OBS_RAD + 0.05)
        if np.any(obs_near):
            s0 = np.where(np.diff(obs_near.astype(int)) ==  1)[0]
            e0 = np.where(np.diff(obs_near.astype(int)) == -1)[0]
            if obs_near[0]:  s0 = np.concatenate([[0], s0])
            if obs_near[-1]: e0 = np.concatenate([e0, [len(obs_near)-1]])
            for i, (s, e) in enumerate(zip(s0, e0)):
                axi.axvspan(t[s], t[e], alpha=0.10, color="#AAAAAA",
                            label="Near obstacle" if i == 0 else None, zorder=0)

        for tv in [T_CARRY_END, T_HOLD_END]:
            axi.axvline(tv, color="#888888", lw=0.8, ls="--", alpha=0.45)

    ax0, ax1 = axes

    ax0.plot(t, pos[:,0], color=c_x, lw=1.8, label=r"$x(t)$")
    ax0.plot(t, pos[:,1], color=c_y, lw=1.8, label=r"$y(t)$")
    ax0.plot(t, pos[:,2], color=c_z, lw=1.8, label=r"$z(t)$")
    for val, col, ls in [
        (START[0], c_x, ":"), (START[1], c_y, ":"), (START[2], c_z, ":"),
        (WAYPOINT[0], c_x, "-."), (WAYPOINT[1], c_y, "-."),
        (GOAL[0], c_x, "--"), (GOAL[1], c_y, "--"), (GOAL[2], c_z, "--"),
    ]:
        ax0.axhline(val, color=col, lw=0.6, ls=ls, alpha=0.30)
    ylo0, yhi0 = pos.min() - 0.03, pos.max() + 0.04
    ax0.set_ylim(ylo0, yhi0)
    ax0.set_ylabel("Position (m)", fontsize=11)
    ax0.legend(fontsize=8.5, loc="upper right", framealpha=0.9,
               edgecolor="lightgrey", fancybox=False, ncol=3)
    yhi_l = ax0.get_ylim()[1]
    for tx, lb, col in [(2.5, "Carry", C_CARRY), (6.0, "Hold", C_HOLD), (9.0, "Continue", C_CONT)]:
        ax0.text(tx, yhi_l * 0.97, lb, fontsize=7.5, color=col,
                 ha="center", fontweight="bold", alpha=0.65)

    # distance traces
    ax0_r = ax0.twinx()
    ax0_r.plot(t, d_obs,  color="#999999", lw=1.0, ls=":", alpha=0.55, label="d(obstacle)")
    ax0_r.plot(t, d_wp,   color=C_WP,      lw=1.0, ls=":", alpha=0.55, label="d(waypoint)")
    ax0_r.plot(t, d_goal, color=C_GOAL,    lw=1.0, ls=":", alpha=0.55, label="d(goal)")
    ax0_r.axhline(OBS_SAFE_RAD, color="#333333", lw=0.7, ls="-", alpha=0.40,
                  label=f"safe r={OBS_SAFE_RAD:.2f} m")
    ax0_r.set_ylabel("Distance (m)", fontsize=9, color="#777777")
    ax0_r.tick_params(labelsize=8)
    ax0_r.legend(fontsize=7.5, loc="center right", framealpha=0.85,
                 edgecolor="lightgrey", fancybox=False)

    # velocity panel
    ax1.plot(t, vel[:,0], color=c_x, lw=1.5, alpha=0.75, label=r"$\dot x$")
    ax1.plot(t, vel[:,1], color=c_y, lw=1.5, alpha=0.75, label=r"$\dot y$")
    ax1.plot(t, vel[:,2], color=c_z, lw=1.5, alpha=0.75, label=r"$\dot z$")
    ax1.plot(t, speed,    color=c_spd, lw=2.2, label=r"$\|\dot p\|$")
    vc = trace.safety.get("velocity_cbf", {}) if hasattr(trace, "safety") else {}
    vmax_active = vc.get("vmax_active", None)
    window_active = vc.get("window_active", None)
    if vmax_active is not None and len(vmax_active) == len(t):
        ax1.plot(t, vmax_active, color="#333333", ls=":", lw=1.2, alpha=0.75,
                 label=r"$v_{max}(t)$")
        ax1.plot(t, -vmax_active, color="#333333", ls=":", lw=1.2, alpha=0.75)
    else:
        ax1.axhline( VELOCITY_LIMIT, color="#333333", ls=":", lw=1.2, alpha=0.65,
                    label=fr"$v_{{max}}$={VELOCITY_LIMIT:.2f}")
        ax1.axhline(-VELOCITY_LIMIT, color="#333333", ls=":", lw=1.2, alpha=0.65)
    ax1.axhline( 0.0, color="#999999", ls="-", lw=0.5, alpha=0.30)

    # hold phase: velocity should be ~0
    ax1.axhspan(-0.06, 0.06, alpha=0.08, color=C_HOLD,
                label="Target: v≈0 in hold")

    vhi = max(speed.max(), VELOCITY_LIMIT, 0.05) * 1.20
    ax1.set_ylim(-vhi, vhi)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Velocity (m/s)", fontsize=11)
    ax1.set_xlim(0, T_CONT_END)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.9,
               edgecolor="lightgrey", fancybox=False, ncol=3)

    fig.suptitle("Exp 2 — Per-axis Position & Velocity", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# =====================================================================
#  PLOT 6 — angular velocity vs time
# =====================================================================
def plot_angular_velocity(trace, best_cost, base="exp2_angular_velocity"):
    if trace.angular_velocity is None:
        print("No angular velocity — skipping.")
        return
    t = trace.time
    omega = trace.angular_velocity
    omega_norm = np.linalg.norm(omega, axis=1)

    av = trace.safety.get("angular_velocity_cbf", {}) if hasattr(trace, "safety") else {}
    omega_max = av.get("omega_max", None)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(0,            T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
    ax.axvspan(T_CARRY_END,  T_HOLD_END,  alpha=0.040, color=C_HOLD,  zorder=0)
    ax.axvspan(T_HOLD_END,   T_CONT_END,  alpha=0.025, color=C_CONT,  zorder=0)

    ax.plot(t, omega[:,0], color=C_KX, lw=1.4, alpha=0.70, label=r"$\omega_x$")
    ax.plot(t, omega[:,1], color=C_KY, lw=1.4, alpha=0.70, label=r"$\omega_y$")
    ax.plot(t, omega[:,2], color=C_KZ, lw=1.4, alpha=0.70, label=r"$\omega_z$")
    ax.plot(t, omega_norm, color="#8172B2", lw=2.2, label=r"$\|\omega\|$")

    if omega_max is not None and np.isfinite(omega_max):
        ax.axhline(omega_max, color="#333333", ls=":", lw=1.2, alpha=0.75,
                   label=fr"$\omega_{{max}}$={omega_max:.2f}")

    for tv in [T_CARRY_END, T_HOLD_END]:
        ax.axvline(tv, color="#888888", lw=0.8, ls="--", alpha=0.50)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Angular velocity (rad/s)", fontsize=11)
    ax.set_title(f"{SCENE_LABEL}\nAngular velocity", fontsize=9)
    ax.set_xlim(0, T_CONT_END)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# =====================================================================
#  PLOT 7 — velocity CBF windows (speed vs vmax(t))
# =====================================================================
def plot_velocity_cbf(trace, best_cost, base="exp2_velocity_cbf"):
    vc = trace.safety.get("velocity_cbf", {}) if hasattr(trace, "safety") else {}
    if not vc:
        print("No velocity CBF data — skipping.")
        return
    t = trace.time
    speed = np.linalg.norm(trace.velocity, axis=1)
    vmax_active = vc.get("vmax_active", None)
    window_active = vc.get("window_active", None)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(0,            T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
    ax.axvspan(T_CARRY_END,  T_HOLD_END,  alpha=0.040, color=C_HOLD,  zorder=0)
    ax.axvspan(T_HOLD_END,   T_CONT_END,  alpha=0.025, color=C_CONT,  zorder=0)

    ax.plot(t, speed, color="#8172B2", lw=2.2, label=r"$\|\dot p\|$")
    if vmax_active is not None and len(vmax_active) == len(t):
        ax.plot(t, vmax_active, color="#333333", ls=":", lw=1.2, alpha=0.80,
                label=r"$v_{max}(t)$")
    else:
        ax.axhline(VELOCITY_LIMIT, color="#333333", ls=":", lw=1.2, alpha=0.65,
                   label=fr"$v_{{max}}$={VELOCITY_LIMIT:.2f}")

    if window_active is not None and len(window_active) == len(t):
        on = window_active.astype(bool)
        if np.any(on):
            ax.fill_between(t, 0.0, speed.max() * 1.05, where=on,
                            color="#CCCCCC", alpha=0.18, label="CBF window on")

    for tv in [T_CARRY_END, T_HOLD_END]:
        ax.axvline(tv, color="#888888", lw=0.8, ls="--", alpha=0.50)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Speed (m/s)", fontsize=11)
    ax.set_title(f"{SCENE_LABEL}\nVelocity CBF windows", fontsize=9)
    ax.set_xlim(0, T_CONT_END)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# =====================================================================
#  PLOT 8 — linear and angular acceleration
# =====================================================================
def plot_acceleration(trace, best_cost, base="exp2_acceleration"):
    t = trace.time
    if trace.velocity is None:
        print("No velocity — skipping acceleration plot.")
        return
    vel = trace.velocity
    acc = np.gradient(vel, t, axis=0)
    acc_norm = np.linalg.norm(acc, axis=1)

    if trace.angular_velocity is not None:
        omega = trace.angular_velocity
        alpha = np.gradient(omega, t, axis=0)
        alpha_norm = np.linalg.norm(alpha, axis=1)
    else:
        alpha = None
        alpha_norm = None

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax in axes:
        ax.axvspan(0,            T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
        ax.axvspan(T_CARRY_END,  T_HOLD_END,  alpha=0.040, color=C_HOLD,  zorder=0)
        ax.axvspan(T_HOLD_END,   T_CONT_END,  alpha=0.025, color=C_CONT,  zorder=0)
        for tv in [T_CARRY_END, T_HOLD_END]:
            ax.axvline(tv, color="#888888", lw=0.8, ls="--", alpha=0.50)

    ax0, ax1 = axes
    ax0.plot(t, acc[:,0], color=C_KX, lw=1.4, alpha=0.75, label=r"$\ddot x$")
    ax0.plot(t, acc[:,1], color=C_KY, lw=1.4, alpha=0.75, label=r"$\ddot y$")
    ax0.plot(t, acc[:,2], color=C_KZ, lw=1.4, alpha=0.75, label=r"$\ddot z$")
    ax0.plot(t, acc_norm, color="#8172B2", lw=2.0, label=r"$\|\ddot p\|$")
    ax0.axhline(0.0, color="#999999", ls="-", lw=0.5, alpha=0.30)
    ax0.set_ylabel("Linear accel (m/s²)", fontsize=11)
    ax0.set_title(f"{SCENE_LABEL}\nLinear & angular acceleration", fontsize=9)
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=9, loc="upper right", framealpha=0.9,
               edgecolor="lightgrey", fancybox=False, ncol=2)

    if alpha is not None:
        ax1.plot(t, alpha[:,0], color=C_KX, lw=1.4, alpha=0.75, label=r"$\dot\omega_x$")
        ax1.plot(t, alpha[:,1], color=C_KY, lw=1.4, alpha=0.75, label=r"$\dot\omega_y$")
        ax1.plot(t, alpha[:,2], color=C_KZ, lw=1.4, alpha=0.75, label=r"$\dot\omega_z$")
        ax1.plot(t, alpha_norm, color="#8172B2", lw=2.0, label=r"$\|\dot\omega\|$")
        ax1.axhline(0.0, color="#999999", ls="-", lw=0.5, alpha=0.30)
        ax1.set_ylabel("Angular accel (rad/s²)", fontsize=11)
        ax1.grid(True, alpha=0.25)
        ax1.legend(fontsize=9, loc="upper right", framealpha=0.9,
                   edgecolor="lightgrey", fancybox=False, ncol=2)
    else:
        ax1.text(0.5, 0.5, "No angular velocity", transform=ax1.transAxes,
                 ha="center", va="center", fontsize=10, color="#666666")

    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_xlim(0, T_CONT_END)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# =====================================================================
#  PLOT 10 — obstacle HOCBF diagnostics
# =====================================================================
def plot_obstacle_hocbf(trace, best_cost, base="exp2_obstacle_hocbf"):
    ob = trace.safety.get("obstacle_hocbf", {}) if hasattr(trace, "safety") else {}
    if not ob:
        print("No obstacle HOCBF data — skipping.")
        return
    t = trace.time
    h = np.asarray(ob.get("h", []), dtype=float)
    hdot = np.asarray(ob.get("hdot", []), dtype=float)
    active = np.asarray(ob.get("active", []), dtype=bool)
    proj = trace.safety.get("obstacle_projection", {}) if hasattr(trace, "safety") else {}
    proj_active = np.asarray(proj.get("active", []), dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax in axes:
        ax.axvspan(0,            T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
        ax.axvspan(T_CARRY_END,  T_HOLD_END,  alpha=0.040, color=C_HOLD,  zorder=0)
        ax.axvspan(T_HOLD_END,   T_CONT_END,  alpha=0.025, color=C_CONT,  zorder=0)
        for tv in [T_CARRY_END, T_HOLD_END]:
            ax.axvline(tv, color="#888888", lw=0.8, ls="--", alpha=0.50)

    ax0, ax1 = axes
    ax0.plot(t, h, color="#4C72B0", lw=2.0, label=r"$h(x)$")
    ax0.axhline(0.0, color="#CC4444", ls=":", lw=1.0, alpha=0.80,
                label=r"$h=0$ (boundary)")
    ax0.set_ylabel("$h(x)$", fontsize=11)
    ax0.set_title(f"{SCENE_LABEL}\nObstacle HOCBF diagnostics", fontsize=9)
    ax0.grid(True, alpha=0.25)

    ax1.plot(t, hdot, color="#DD8452", lw=1.8, label=r"$\dot h$")
    ax1.axhline(0.0, color="#999999", ls="-", lw=0.5, alpha=0.30)
    hdot_lo = float(np.nanmin(hdot)) if hdot.size and np.any(np.isfinite(hdot)) else -1.0
    hdot_hi = float(np.nanmax(hdot)) if hdot.size and np.any(np.isfinite(hdot)) else 1.0
    if len(active) == len(t) and np.any(active):
        ax1.fill_between(t, hdot_lo, hdot_hi,
                         where=active, color="#AAAAAA", alpha=0.18,
                         label="HOCBF active")
    if len(proj_active) == len(t) and np.any(proj_active):
        ax1.fill_between(t, hdot_lo, hdot_hi,
                         where=proj_active, color="#7F7F7F", alpha=0.18,
                         label="Projection active")
    ax1.set_ylabel(r"$\dot h$", fontsize=11)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_xlim(0, T_CONT_END)
    ax1.grid(True, alpha=0.25)

    for ax in axes:
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  framealpha=0.9, edgecolor="lightgrey", fancybox=False)

    fig.subplots_adjust(right=0.78)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ======================================================================
#  CSV export
# ======================================================================
def save_trajectory_csv(trace, csv_path="exp2_trajectory.csv"):
    import csv
    pos   = trace.position
    T     = len(trace.time)
    if trace.orientation is not None and len(trace.orientation) == T:
        quat = trace.orientation
    else:
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
    K_arr = trace.gains["K"]
    D_arr = trace.gains["D"]
    header = [
        "x","y","z","qw","qx","qy","qz",
        "k11","k12","k13","k21","k22","k23","k31","k32","k33",
        "d11","d12","d13","d21","d22","d23","d31","d32","d33",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(T):
            K = K_arr[i]; D = D_arr[i]
            row = [
                f"{pos[i,0]:.8f}", f"{pos[i,1]:.8f}", f"{pos[i,2]:.8f}",
                f"{quat[i,0]:.8f}", f"{quat[i,1]:.8f}", f"{quat[i,2]:.8f}", f"{quat[i,3]:.8f}",
                f"{K[0,0]:.8f}", f"{K[0,1]:.8f}", f"{K[0,2]:.8f}",
                f"{K[1,0]:.8f}", f"{K[1,1]:.8f}", f"{K[1,2]:.8f}",
                f"{K[2,0]:.8f}", f"{K[2,1]:.8f}", f"{K[2,2]:.8f}",
                f"{D[0,0]:.8f}", f"{D[0,1]:.8f}", f"{D[0,2]:.8f}",
                f"{D[1,0]:.8f}", f"{D[1,1]:.8f}", f"{D[1,2]:.8f}",
                f"{D[2,0]:.8f}", f"{D[2,1]:.8f}", f"{D[2,2]:.8f}",
            ]
            writer.writerow(row)
    print(f"Saved: {csv_path}  ({T} rows)")


# ======================================================================
#  main
# ======================================================================
def main():
    taskspec = load_taskspec_from_json("spec/exp2_task.json")
    assert taskspec.phases is not None

    global START, WAYPOINT, GOAL, OBSTACLE, OBS_RAD, OBS_SAFE_RAD
    global Q_UPRIGHT, ORI_REF, ORI_MAX_ANGLE, T_CARRY_END, T_HOLD_END, T_CONT_END
    global HUMAN_POS, OBSTACLE_GEOMETRY, VELOCITY_LIMIT

    START = np.asarray(taskspec.phases[0]["start"], dtype=float)
    GOAL = np.asarray(taskspec.phases[-1]["end"], dtype=float)

    at_waypoint_clause = next((cl for cl in taskspec.clauses if cl.predicate == "AtWaypoint"), None)
    if at_waypoint_clause is not None and "waypoint" in at_waypoint_clause.parameters:
        WAYPOINT = np.asarray(at_waypoint_clause.parameters["waypoint"], dtype=float)
    else:
        WAYPOINT = np.asarray(taskspec.phases[0]["end"], dtype=float)

    phase_durations = [float(p["duration"]) for p in taskspec.phases]
    T_CARRY_END = phase_durations[0]
    T_HOLD_END = phase_durations[0] + phase_durations[1] if len(phase_durations) > 1 else phase_durations[0]
    T_CONT_END = float(np.sum(phase_durations))

    Q_UPRIGHT = quat_normalize(np.asarray(
        taskspec.phases[0].get("start_quat", [1.0, 0.0, 0.0, 0.0]), dtype=float
    ))

    ori_clause = next((cl for cl in taskspec.clauses if cl.predicate == "OrientationLimit"), None)
    if ori_clause is not None and "q_ref" in ori_clause.parameters:
        ORI_REF = quat_normalize(np.asarray(ori_clause.parameters["q_ref"], dtype=float))
        ORI_MAX_ANGLE = float(ori_clause.parameters.get("max_angle_rad", ORI_MAX_ANGLE))
    else:
        ORI_REF = Q_UPRIGHT.copy()

    obs_clause = next((cl for cl in taskspec.clauses if cl.predicate == "ObstacleAvoidance"), None)
    if obs_clause is None:
        raise ValueError("Exp2 requires an ObstacleAvoidance clause in spec/exp2_task.json")
    OBSTACLE = np.asarray(obs_clause.parameters["obstacle_position"], dtype=float)
    OBS_RAD = float(obs_clause.parameters["safe_radius"])
    OBS_SAFE_RAD = OBS_RAD
    OBSTACLE_GEOMETRY = str(obs_clause.parameters.get("geometry", "sphere"))

    vel_clause = next((cl for cl in taskspec.clauses if cl.predicate == "VelocityLimit"), None)
    if vel_clause is not None and "vmax" in vel_clause.parameters:
        VELOCITY_LIMIT = float(vel_clause.parameters["vmax"])

    HUMAN_POS = GOAL.copy()

    policy = MultiPhaseCertifiedPolicy(taskspec.phases, K0=300.0, D0=30.0)

    # Layers 1+2 (DMP repulsion + radial projector) are wired automatically
    # from the JSON modality="HARD" clause — no manual hardcoding needed.
    policy.setup_hard_obstacles_from_taskspec(taskspec)

    theta_dim = policy.parameter_dimension()
    print(f"Exp 2: Carry → Hold-2s → Continue (HARD obstacle avoidance, from JSON)")
    print(f"Multi-phase policy: {len(taskspec.phases)} phases, theta_dim={theta_dim}")
    print(f"  obstacle geometry: {OBSTACLE_GEOMETRY}, radius={OBS_RAD:.3f} m")
    print(f"  velocity limit: HARD CBF vmax={VELOCITY_LIMIT:.3f} m/s")
    print(f"  has_orientation: {policy.has_orientation}")
    print(f"  orientation ref: {ORI_REF}  max_angle={ORI_MAX_ANGLE:.3f} rad")
    for i, p in enumerate(taskspec.phases):
        od = policy.ori_dims[i]
        print(f"  Phase {i+1} ({p['label']}): pos_dim={policy.theta_dims[i]-od}, ori_dim={od}")

    predicate_registry = build_predicate_registry()
    compiler = Compiler(predicate_registry,
                        human_position=None,
                        human_proximity_radius=None)
    objective_fn = compiler.compile(taskspec)

    trace0 = policy.rollout(np.zeros(theta_dim))
    cost0  = objective_fn(trace0)
    print(f"Nominal cost (theta=0): {cost0:.4f}")

    theta_init = np.zeros(theta_dim)
    sigma_init = policy.structured_sigma(
        sigma_traj_xy=3.0,
        sigma_traj_z=0.5,
        sigma_sd=2.0,
        sigma_sk=2.0,
        sigma_ori=1.5,
    )
    # Dampen hold phase position noise — hold must stay still
    hold_off = policy.offsets[1]
    hold_dim = policy.theta_dims[1]
    sigma_init[hold_off:hold_off + hold_dim] *= 0.03

    optimizer = PIBB(theta=theta_init, sigma=sigma_init, beta=8.0, decay=0.99)

    N_SAMPLES = 30
    N_UPDATES = 70
    best_cost  = float("inf")
    best_theta = theta_init.copy()

    hard_strength = 0.05
    hard_infl = 2.5
    if obs_clause.hard_obstacle is not None:
        hard_strength = float(obs_clause.hard_obstacle.get("strength", hard_strength))
        hard_infl = float(obs_clause.hard_obstacle.get("infl_factor", hard_infl))

    print(f"\nPIBB: {N_UPDATES} updates x {N_SAMPLES} samples")
    print(
        f"  Obstacle: HARD  (geometry={OBSTACLE_GEOMETRY}, "
        f"DMP repulsion str={hard_strength:.3f}, infl={hard_infl:.2f}, "
        f"projector r={OBS_SAFE_RAD:.2f} m)"
    )

    for upd in range(N_UPDATES):
        samples = optimizer.sample(N_SAMPLES)
        costs   = np.array([objective_fn(policy.rollout(samples[i]))
                            for i in range(N_SAMPLES)])
        costs_s = np.clip(np.where(np.isfinite(costs), costs, 1e4), 0.0, 1e4)
        optimizer.update(samples, costs_s)

        if costs.min() < best_cost:
            best_cost  = costs.min()
            best_theta = samples[np.argmin(costs)].copy()

        if (upd + 1) % 1 == 0 or upd == 0:
            print(f"  [{upd+1:03d}]  min={costs.min():.4f}  "
                  f"mean={costs.mean():.4f}  best={best_cost:.4f}")

    print("Optimization complete.\n")

    trace_final = policy.rollout(best_theta)
    save_trajectory_csv(trace_final, csv_path="exp2_trajectory.csv")
    print_diagnostics(trace_final, best_cost)

    plot_3d_workspace(trace_final, best_cost)
    plot_2d_topdown(trace_final, best_cost)
    plot_stiffness(trace_final, best_cost)
    plot_orientation_euler(trace_final, best_cost)
    plot_orientation_geodesic(trace_final, best_cost)
    plot_kinematics(trace_final, best_cost)
    plot_angular_velocity(trace_final, best_cost)
    plot_velocity_cbf(trace_final, best_cost)
    plot_acceleration(trace_final, best_cost)
    plot_obstacle_hocbf(trace_final, best_cost)

    print("Exp 2 done — 10 plots saved as PNG.")


if __name__ == "__main__":
    main()
