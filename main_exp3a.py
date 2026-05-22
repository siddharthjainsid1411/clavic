"""
Exp 3a: Carry-then-Pour with Human-Proximity Per-Axis Stiffness Reduction.

2-phase task (no separate hold):
  Phase 1 (0-7 s) : carry mug upright from start to goal, avoid obstacle
    Phase 2 (7-10 s): pour -- tilt 90 deg about Y axis, hold position at goal
    Geometry and orientations are loaded from spec/exp3a_task.json
    Stiffness reduction near the human is handled by the compiler penalty
CGMS guarantee K = Q^T Q > 0 is maintained throughout.

3 publication-quality plots (PDF + PNG, 300 dpi):
  1. scene3_workspace.pdf   -- Franka FRS 3D view (X=forward, Y=lateral, Z=height)
  2. scene3_stiffness.pdf   -- Per-axis Kxx/Kyy/Kzz vs time
  3. scene3_orientation.pdf -- Euler pitch (pour axis) vs time; roll/yaw near zero
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns

from spec.compiler import Compiler
from spec.json_parser import load_taskspec_from_json
from core.multi_phase_policy import MultiPhaseCertifiedPolicy
from optimization.optimizer import PIBB
from core.cgms.quat_utils import quat_distance, quat_normalize

from logic.predicates import (
    at_goal_pose, at_waypoint, hold_at_waypoint,
    obstacle_avoidance, velocity_limit,
    orientation_at_target, orientation_limit,
    angular_velocity_limit,
)

# ── seaborn style ──────────────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# ── scene constants (loaded from JSON in main()) ──────────────────────
START    = None
GOAL     = None
OBSTACLE = None
OBS_RAD  = 0.12
OBSTACLE_GEOMETRY = "sphere"

Q_UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])
Q_POUR    = quat_normalize(np.array([0.70710678, 0.0, 0.70710678, 0.0]))

# 2-phase timing (loaded from JSON in main())
T_CARRY_END = 7.0
T_POUR_END  = 10.0

HUMAN_POS      = np.zeros(3)
HUMAN_PROX_RAD = 0.12
HUMAN_RAMP_RAD = 0.36        # ramp starts at 3× proximity radius (matches compiler RAMP_FACTOR=3)
K_AXIS_LIMIT   = 100.0       # N/m per axis — target inside proximity radius (matches compiler)

# obstacle box half-extents for plot
OBS_HX, OBS_HY, OBS_HZ = 0.08, 0.08, 0.05

# ── colours ───────────────────────────────────────────────────────────
C_CARRY  = "#4C72B0"
C_POUR   = "#C44E52"
C_OBS    = "#AAAAAA"
C_HUMAN  = "#AEC7E8"
C_START  = "#2CA02C"
C_GOAL   = "#1F77B4"
C_DASH   = "#999999"
C_KX     = "#4C72B0"
C_KY     = "#DD8452"
C_KZ     = "#55A868"


# ── predicates ────────────────────────────────────────────────────────
def build_predicate_registry():
    return {
        "AtGoalPose":           at_goal_pose,
        "AtWaypoint":           at_waypoint,
        "HoldAtWaypoint":       hold_at_waypoint,
        "ObstacleAvoidance":    obstacle_avoidance,
        "VelocityLimit":        velocity_limit,
        "OrientationAtTarget":  orientation_at_target,
        "OrientationLimit":     orientation_limit,
        "AngularVelocityLimit": angular_velocity_limit,
    }


# ── quaternion -> Euler ZYX (degrees) ────────────────────────────────
def quat_to_euler(q):
    """Returns (roll_x, pitch_y, yaw_z) in degrees."""
    w, x, y, z = q
    roll  = np.degrees(np.arctan2(2*(w*x + y*z),  1 - 2*(x*x + y*y)))
    sinp  = np.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y),  1 - 2*(y*y + z*z)))
    return roll, pitch, yaw


# ── diagnostics ───────────────────────────────────────────────────────
def print_diagnostics(trace, best_cost):
    pos    = trace.position
    speed  = np.linalg.norm(trace.velocity, axis=1)
    K_arr  = trace.gains["K"]
    D_arr  = trace.gains["D"]
    trK    = np.array([np.trace(K) for K in K_arr])
    trD    = np.array([np.trace(D) for D in D_arr])
    K_diag = np.array([[K[0,0], K[1,1], K[2,2]] for K in K_arr])

    d_goal  = np.linalg.norm(pos - GOAL, axis=1)
    reached = bool(np.any(d_goal < 0.05))
    t_reach = float(trace.time[np.argmax(d_goal < 0.05)]) if reached else -1.0

    if OBSTACLE_GEOMETRY == "cylinder_infinite":
        d_obs = np.linalg.norm(pos[:, :2] - OBSTACLE[:2], axis=1)
    else:
        d_obs = np.linalg.norm(pos - OBSTACLE, axis=1)
    obs_cm    = (d_obs.min() - OBS_RAD) * 100.0
    K_eig_min = float(min(np.linalg.eigvalsh(K)[0] for K in K_arr))

    # How many carry-phase timesteps were projected to the hard boundary.
    carry_mask_d = trace.time <= T_CARRY_END
    pos_carry_d  = pos[carry_mask_d]
    if OBSTACLE_GEOMETRY == "cylinder_infinite":
        d_carry_obs = np.linalg.norm(pos_carry_d[:, :2] - OBSTACLE[:2], axis=1)
    else:
        d_carry_obs = np.linalg.norm(pos_carry_d - OBSTACLE, axis=1)
    n_projected  = int(np.sum(np.abs(d_carry_obs - OBS_RAD) < 1e-3))

    near_mask = d_goal < HUMAN_PROX_RAD
    ramp_mask = d_goal < HUMAN_RAMP_RAD

    pour_mask  = trace.time > T_CARRY_END
    pour_drift = float(np.linalg.norm(pos[pour_mask] - GOAL, axis=1).max()) \
                 if np.any(pour_mask) else 0.0

    sep = "=" * 48
    print(f"\n{sep} EXP 3a DIAGNOSTICS {sep}")
    print(f"  Best cost          : {best_cost:.4f}")
    print(f"  Goal reached       : {'YES' if reached else 'NO'}  t={t_reach:.2f} s")
    print(f"  Max speed          : {speed.max():.4f} m/s  (limit 0.8)")
    print(f"  Obstacle clearance : {obs_cm:.1f} cm  (hard {OBSTACLE_GEOMETRY}, r={OBS_RAD:.2f} m)")
    print(f"  Pts on proj surface: {n_projected}  (0 = DMP already avoids; >0 = projector active)")
    print(f"  Pour pos drift     : {pour_drift*100:.1f} cm  (target < 5)")
    print(f"  tr(K) range        : [{trK.min():.0f}, {trK.max():.0f}] N/m")
    print(f"  tr(D) range        : [{trD.min():.1f}, {trD.max():.1f}] Ns/m")
    print(f"  K eigenvalue min   : {K_eig_min:.4f}  (CGMS > 0 required)")

    if np.any(ramp_mask):
        kr = K_diag[ramp_mask]
        print(f"  Ramp-zone K (d < {HUMAN_RAMP_RAD} m, penalty starts):")
        for i, ax in enumerate(["xx", "yy", "zz"]):
            print(f"    K_{ax}: [{kr[:,i].min():.0f}, {kr[:,i].max():.0f}] N/m")

    if np.any(near_mask):
        kn = K_diag[near_mask]
        print(f"  Hard-zone K (d < {HUMAN_PROX_RAD} m, full penalty):")
        for i, ax in enumerate(["xx", "yy", "zz"]):
            print(f"    K_{ax}: [{kn[:,i].min():.0f}, {kn[:,i].max():.0f}] N/m  (target < {K_AXIS_LIMIT:.0f})")

    if trace.orientation is not None:
        carry_mask = trace.time <= T_CARRY_END
        max_tilt   = float(max(quat_distance(trace.orientation[k], Q_UPRIGHT)
                               for k in range(len(trace.time)) if carry_mask[k]))
        err_end    = float(quat_distance(trace.orientation[-1], Q_POUR))
        omega_max  = float(np.linalg.norm(trace.angular_velocity, axis=1).max())
        print(f"  Max tilt carry     : {np.degrees(max_tilt):.1f} deg  (limit 15)")
        print(f"  Pour error (end)   : {np.degrees(err_end):.1f} deg  (target < 10)")
        print(f"  Max angular vel    : {omega_max:.3f} rad/s  (limit 1.5)")

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

    rep = trace.safety.get("repulsive_force", {}) if hasattr(trace, "safety") else {}
    if rep:
        rep_active = rep.get("active", [])
        print(f"  Repulsion active   : {int(np.sum(rep_active))} steps")
        print(f"  Repulsion windows  : {_mask_intervals(trace.time, rep_active)}")

    ob = trace.safety.get("obstacle_hocbf", {}) if hasattr(trace, "safety") else {}
    if ob:
        print(f"  Obstacle HOCBF     : {_mask_intervals(trace.time, ob.get('active', []))}")

    vc = trace.safety.get("velocity_cbf", {}) if hasattr(trace, "safety") else {}
    if vc:
        print(f"  Velocity CBF       : {_mask_intervals(trace.time, vc.get('active', []))}")

    oh = trace.safety.get("orientation_hocbf", {}) if hasattr(trace, "safety") else {}
    if oh:
        print(f"  Orientation HOCBF  : {_mask_intervals(trace.time, oh.get('active', []))}")

    av = trace.safety.get("angular_velocity_cbf", {}) if hasattr(trace, "safety") else {}
    if av:
        print(f"  AngVel CBF         : {_mask_intervals(trace.time, av.get('active', []))}")
    print(f"{sep * 2}\n")


# ======================================================================
#  HELPER: remap world coords to Franka-style plot axes
#   X-Z = table plane (bottom floor of 3D plot)
#   Y   = height (vertical axis in 3D plot = matplotlib Z)
#
#   world [x, y, z] -> plot [x=depth, y=lateral(table), z=height]
#   i.e.  matplotlib X = world x,  matplotlib Y = world y,  matplotlib Z = world z
#   (identity mapping — Franka world already has Z as height)
# ======================================================================
def to_plot(p):
    """p: (..., 3) world [x,y,z] -> plot [x, y, z] (identity — Z is height/vertical)"""
    p = np.asarray(p)
    return np.array(p)   # identity; matplotlib Z = world z = height


def _obs_box_faces(cx, cy, cz, dx, dy, dz_bot, dz_top):
    """Return 6 face lists for a box from (cx±dx, cy±dy, cz+dz_bot) to (cz+dz_top)."""
    corners = np.array([
        [cx-dx, cy-dy, cz+dz_bot], [cx+dx, cy-dy, cz+dz_bot],
        [cx+dx, cy+dy, cz+dz_bot], [cx-dx, cy+dy, cz+dz_bot],
        [cx-dx, cy-dy, cz+dz_top], [cx+dx, cy-dy, cz+dz_top],
        [cx+dx, cy+dy, cz+dz_top], [cx-dx, cy+dy, cz+dz_top],
    ])
    return [
        [corners[0], corners[1], corners[5], corners[4]],  # front
        [corners[2], corners[3], corners[7], corners[6]],  # back
        [corners[0], corners[3], corners[7], corners[4]],  # left
        [corners[1], corners[2], corners[6], corners[5]],  # right
        [corners[4], corners[5], corners[6], corners[7]],  # top
        [corners[0], corners[1], corners[2], corners[3]],  # bottom
    ]


# ======================================================================
#  PLOT 1 -- 3D workspace  (Franka frame: X=depth, Y=height, Z=table)
# ======================================================================
def plot_3d_workspace(trace, best_cost, base="exp3a_workspace"):
    pos = trace.position     # (T, 3)  world coords
    t   = trace.time

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # ── obstacle cuboid: bottom 20% solid black (laptop body), top 80% translucent ──
    # The obstacle IS the laptop on the table.
    # Total height = 2 × OBS_HZ.  Bottom 20% = solid dark base, top 80% = presence zone.
    cx, cy, cz = OBSTACLE
    dx, dy, dz = OBS_HX, OBS_HY, OBS_HZ
    split_z = -dz + 2 * dz * 0.20   # offset from cz: bottom 20% boundary = cz + (-dz + 0.4*dz)
    # Solid base layer (bottom 20% of obstacle height)
    ax.add_collection3d(Poly3DCollection(
        _obs_box_faces(cx, cy, cz, dx, dy, -dz, split_z),
        alpha=0.90, facecolor="#1A1A1A", edgecolor="#000000", linewidth=0.7
    ))
    # Translucent upper layer (top 80%)
    ax.add_collection3d(Poly3DCollection(
        _obs_box_faces(cx, cy, cz, dx, dy, split_z, +dz),
        alpha=0.20, facecolor=C_OBS, edgecolor="#666666", linewidth=0.5
    ))

    # ── human proximity: vertical cylinder (axis along Z = height in plot) ──
    # Table plane is X-Y (matplotlib X=world_x, matplotlib Y=world_y).
    # Height is matplotlib Z = world_z.
    # Cylinder stands upright: its circular cross-section is in the X-Y plane,
    # axis runs along Z (height).
    r_cyl   = HUMAN_PROX_RAD * 0.6       # 0.6× proximity radius
    h_cyl   = 0.35                        # cylinder height (m), looks like human torso
    gx, gy, gz = to_plot(GOAL)            # goal = human position in plot frame
    # bottom and top of cylinder in Z
    z_bot = gz - 0.05                     # slightly below goal height
    z_top = z_bot + h_cyl

    theta_c  = np.linspace(0, 2*np.pi, 40)
    xc = gx + r_cyl * np.cos(theta_c)
    yc = gy + r_cyl * np.sin(theta_c)

    # top/bottom rings
    for zb in [z_bot, z_top]:
        ax.plot(xc, yc, [zb]*len(theta_c),
                color=C_HUMAN, lw=0.6, alpha=0.45)
    # vertical edges
    for i in range(0, len(theta_c), 4):
        ax.plot([xc[i], xc[i]], [yc[i], yc[i]], [z_bot, z_top],
                color=C_HUMAN, lw=0.5, alpha=0.30)
    # filled side surface
    z_fill    = np.linspace(z_bot, z_top, 2)
    Theta, Zg = np.meshgrid(theta_c, z_fill)
    ax.plot_surface(gx + r_cyl*np.cos(Theta),
                    gy + r_cyl*np.sin(Theta),
                    Zg,
                    alpha=0.10, color=C_HUMAN, edgecolor="none")
    # end caps
    for zb in [z_bot, z_top]:
        ax.plot_surface(
            gx + r_cyl * np.outer(np.cos(theta_c), np.ones(2)),
            gy + r_cyl * np.outer(np.sin(theta_c), np.ones(2)),
            np.full((len(theta_c), 2), zb),
            alpha=0.07, color=C_HUMAN, edgecolor="none")

    # ── shortest straight path ──
    sp = to_plot(START)
    gp = to_plot(GOAL)
    ax.plot([sp[0], gp[0]], [sp[1], gp[1]], [sp[2], gp[2]],
            "--", color=C_DASH, lw=1.8, alpha=0.60, label="Shortest path", zorder=2)

    # ── trajectory coloured by phase (2 phases) ──
    ip = np.searchsorted(t, T_CARRY_END)   # end of carry phase
    pp = to_plot(pos)
    ax.plot(pp[:ip+1, 0], pp[:ip+1, 1], pp[:ip+1, 2],
            color=C_CARRY, lw=2.2, solid_capstyle="round", zorder=5, label="Carry")
    ax.plot(pp[ip:, 0], pp[ip:, 1], pp[ip:, 2],
            color=C_POUR,  lw=2.2, solid_capstyle="round", zorder=5, label="Pour")

    # ── start / goal markers ──
    ax.scatter(sp[0], sp[1], sp[2], s=65, c=C_START, zorder=10,
               depthshade=False, edgecolors="black", linewidth=0.6)
    ax.scatter(gp[0], gp[1], gp[2], s=60, c=C_GOAL, zorder=10,
               depthshade=False, edgecolors="black", linewidth=0.6, marker="D")
    ax.text(sp[0]+0.02, sp[1]-0.01, sp[2]+0.03, "Start",
            fontsize=8, fontweight="bold", color=C_START)

    # obstacle label
    obs_p = to_plot(OBSTACLE)
    ax.text(obs_p[0], obs_p[1]+0.04, obs_p[2]+0.03, "Obstacle",
            fontsize=7, color="#555555", ha="center")

    # ── axes labels (Franka FRS: X=forward, Y=lateral, Z=height/up) ──
    # matplotlib X = world x (forward/depth)
    # matplotlib Y = world y (lateral along table)
    # matplotlib Z = world z (height — vertical axis, UP)
    ax.set_xlabel("X — forward/depth (m)", fontsize=9, labelpad=7)
    ax.set_ylabel("Y — lateral (m)",        fontsize=9, labelpad=7)
    ax.set_zlabel("Z — height (m)",         fontsize=9, labelpad=7)
    ax.tick_params(labelsize=8)
    # view: slight elevation so X-Y table plane is visible at bottom, Z is up
    ax.view_init(elev=20, azim=-55)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("lightgrey")
    ax.grid(True, alpha=0.28)

    # ── legend ──
    extra = [
        mpatches.Patch(facecolor="#1A1A1A", alpha=0.90, edgecolor="#000",
                       label="Obstacle base (solid)"),
        mpatches.Patch(facecolor=C_OBS,     alpha=0.30, edgecolor="#666",
                       label="Obstacle upper (presence zone)"),
        mpatches.Patch(facecolor=C_HUMAN,   alpha=0.25, edgecolor=C_HUMAN,
                       label=f"Human zone (r={r_cyl:.2f} m)"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + extra, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)

    fig.subplots_adjust(right=0.78)
    plt.tight_layout()
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    print(f"Saved: {base}.pdf")
    plt.close()


# ======================================================================
#  PLOT 1b -- 2D top-down view: X-Y table plane
#   Confirms trajectory stays at constant Z (no Z drift).
#   Cleaner than 3D for judging path shape and obstacle clearance.
# ======================================================================
def plot_2d_topdown(trace, best_cost, base="exp3a_topdown"):
    pos = trace.position     # (T, 3)
    t   = trace.time

    fig, ax = plt.subplots(figsize=(7, 6))

    # ── obstacle square footprint + circular model overlay ──
    side_half = max(OBS_HX, OBS_HY)
    obs_square = mpatches.Rectangle(
        (OBSTACLE[0] - side_half, OBSTACLE[1] - side_half),
        2 * side_half,
        2 * side_half,
        facecolor=C_OBS,
        edgecolor="#666666",
        linewidth=1.0,
        alpha=0.35,
        zorder=2,
        label="Obstacle footprint (square)",
    )
    obs_model_circle = plt.Circle((OBSTACLE[0], OBSTACLE[1]), OBS_RAD,
                                  fill=False, color="#666666", linestyle="--",
                                  linewidth=1.2, alpha=0.9, zorder=3,
                                  label=f"Obstacle model circle (r={OBS_RAD:.2f} m)")
    ax.add_patch(obs_square)
    ax.add_patch(obs_model_circle)
    ax.text(OBSTACLE[0], OBSTACLE[1] + side_half + 0.02, "Obstacle",
            fontsize=8, ha="center", color="#555555")

    # ── human proximity zones ──
    ramp_circle = plt.Circle((GOAL[0], GOAL[1]), HUMAN_RAMP_RAD,
                              color=C_HUMAN, alpha=0.08, zorder=1,
                              label=f"Ramp zone (r={HUMAN_RAMP_RAD:.2f} m)")
    hard_circle = plt.Circle((GOAL[0], GOAL[1]), HUMAN_PROX_RAD,
                              color=C_HUMAN, alpha=0.20, zorder=1,
                              label=f"Hard zone (r={HUMAN_PROX_RAD:.2f} m)")
    ax.add_patch(ramp_circle)
    ax.add_patch(hard_circle)

    # ── straight-line path ──
    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]],
            "--", color=C_DASH, lw=1.6, alpha=0.55, zorder=3, label="Shortest path")

    # ── trajectory coloured by phase ──
    ip = np.searchsorted(t, T_CARRY_END)
    ax.plot(pos[:ip+1, 0], pos[:ip+1, 1], color=C_CARRY, lw=2.2,
            solid_capstyle="round", zorder=5, label="Carry")
    ax.plot(pos[ip:, 0],   pos[ip:, 1],   color=C_POUR,  lw=2.2,
            solid_capstyle="round", zorder=5, label="Pour")

    # ── start / goal markers ──
    ax.scatter(START[0], START[1], s=80, c=C_START, zorder=10,
               edgecolors="black", linewidth=0.7, label="Start")
    ax.scatter(GOAL[0],  GOAL[1],  s=70, c=C_GOAL,  zorder=10,
               edgecolors="black", linewidth=0.7, marker="D", label="Human (goal)")
    ax.text(START[0] + 0.01, START[1] - 0.035, "Start",
            fontsize=8, fontweight="bold", color=C_START)
    ax.text(GOAL[0]  + 0.01, GOAL[1]  + 0.015, "Human\n(goal)",
            fontsize=8, fontweight="bold", color=C_GOAL)

    ax.set_xlabel("X — forward/depth (m)", fontsize=11)
    ax.set_ylabel("Y — lateral (m)",        fontsize=11)
    ax.set_title("Top-down view: table plane (X–Y)", fontsize=11)

    # equal aspect so circles look circular
    ax.set_aspect("equal")
    margin = 0.08
    xlo = min(START[0], GOAL[0]) - margin
    xhi = max(START[0], GOAL[0]) + margin
    ylo = min(START[1], GOAL[1]) - margin
    yhi = max(START[1], GOAL[1]) + margin
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)

    fig.subplots_adjust(right=0.78)
    plt.tight_layout()
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    print(f"Saved: {base}.pdf")
    plt.close()


# ======================================================================
#  PLOT 2 -- per-axis stiffness vs time
# ======================================================================
def plot_stiffness(trace, best_cost, base="exp3a_stiffness"):
    K_arr  = trace.gains["K"]
    Kd     = np.array([[K[0,0], K[1,1], K[2,2]] for K in K_arr])
    t      = trace.time
    d_goal = np.linalg.norm(trace.position - GOAL, axis=1)

    # Ramp zone: 3×r_h (penalty ramp starts here)
    ramp_mask = d_goal < HUMAN_RAMP_RAD
    # Hard zone: inside r_h (full penalty)
    hard_mask = d_goal < HUMAN_PROX_RAD

    fig, ax = plt.subplots(figsize=(9, 4.2))

    # Phase background
    ax.axvspan(0,           T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
    ax.axvspan(T_CARRY_END, T_POUR_END,  alpha=0.025, color=C_POUR,  zorder=0)

    # Ramp zone shading (light) — where penalty ramp begins
    if np.any(ramp_mask):
        starts = np.where(np.diff(ramp_mask.astype(int)) ==  1)[0]
        ends   = np.where(np.diff(ramp_mask.astype(int)) == -1)[0]
        if ramp_mask[0]:  starts = np.concatenate([[0], starts])
        if ramp_mask[-1]: ends   = np.concatenate([ends, [len(ramp_mask)-1]])
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(t[s], t[e], alpha=0.08, color=C_HUMAN,
                       label="Ramp zone (penalty onset)" if i == 0 else None, zorder=0)

    # Hard zone shading (stronger) — inside proximity radius
    if np.any(hard_mask):
        starts = np.where(np.diff(hard_mask.astype(int)) ==  1)[0]
        ends   = np.where(np.diff(hard_mask.astype(int)) == -1)[0]
        if hard_mask[0]:  starts = np.concatenate([[0], starts])
        if hard_mask[-1]: ends   = np.concatenate([ends, [len(hard_mask)-1]])
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(t[s], t[e], alpha=0.18, color=C_HUMAN,
                       label=f"Hard zone (d < {HUMAN_PROX_RAD} m)" if i == 0 else None, zorder=0)

    ax.plot(t, Kd[:,0], color=C_KX, lw=2.0, label=r"$K_{xx}$")
    ax.plot(t, Kd[:,1], color=C_KY, lw=2.0, label=r"$K_{yy}$")
    ax.plot(t, Kd[:,2], color=C_KZ, lw=2.0, label=r"$K_{zz}$")

    ax.axhline(K_AXIS_LIMIT, color="#333333", ls=":", lw=1.2, alpha=0.65,
               label=f"$K_{{\\rm limit}}$ = {K_AXIS_LIMIT:.0f} N/m")

    # Dynamic y-top for phase labels
    ax.set_ylim(bottom=0)
    yhi = max(Kd.max() * 1.10, K_AXIS_LIMIT * 1.5)
    ax.set_ylim(0, yhi)
    for tx, lb, col in [(3.5, "Carry", C_CARRY), (8.5, "Pour", C_POUR)]:
        ax.text(tx, yhi * 0.94, lb, fontsize=8, color=col,
                ha="center", fontweight="bold", alpha=0.70)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Stiffness (N/m)", fontsize=11)
    ax.set_xlim(0, T_POUR_END)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False, ncol=2)

    plt.tight_layout()
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    print(f"Saved: {base}.pdf")
    plt.close()


# ======================================================================
#  PLOT 3 -- Euler orientation vs time
#   Pour = 90-deg rotation about Y axis only.
#   Roll (x) and Yaw (z) should stay ~0 throughout.
#   Pitch (y) rises from 0 to ~90 deg during pour phase.
# ======================================================================
def plot_orientation_euler(trace, best_cost, base="exp3a_orientation"):
    if trace.orientation is None:
        print("No orientation -- skipping plot 3.")
        return
    q     = trace.orientation
    t     = trace.time
    euler = np.array([quat_to_euler(q[k]) for k in range(len(t))])
    roll  = euler[:, 0]
    pitch = euler[:, 1]
    yaw   = euler[:, 2]

    fig, ax = plt.subplots(figsize=(9, 4.2))

    # phase background
    ax.axvspan(0,           T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
    ax.axvspan(T_CARRY_END, T_POUR_END,  alpha=0.025, color=C_POUR,  zorder=0)

    # roll and yaw in lighter style (should stay near 0)
    ax.plot(t, roll, color=C_KX, lw=1.4, ls="--", alpha=0.55,
            label="roll  $\\theta_x$ (should be ~0)")
    ax.plot(t, yaw,  color=C_KZ, lw=1.4, ls="--", alpha=0.55,
            label="yaw   $\\theta_z$ (should be ~0)")
    # pitch (pour axis) prominent solid line
    ax.plot(t, pitch, color=C_KY, lw=2.5,
            label="pitch $\\theta_y$ (pour axis)")

    # target lines
    ax.axhline(90.0, color="#333333", ls=":", lw=1.2, alpha=0.55,
               label="Pour target (90 deg)")
    ax.axhline(0.0,  color="#999999", ls="-", lw=0.5, alpha=0.30, zorder=1)

    # y limits: show 0 -> 100 deg range clearly
    all_max = max(abs(roll).max(), abs(pitch).max(), abs(yaw).max(), 95)
    ax.set_ylim(-15, all_max * 1.10)

    yhi = ax.get_ylim()[1]
    for tx, lb, col in [(3.0, "Carry", C_CARRY), (8.0, "Pour", C_POUR)]:
        ax.text(tx, yhi * 0.93, lb, fontsize=8, color=col,
                ha="center", fontweight="bold", alpha=0.70)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Angle (deg)", fontsize=11)
    ax.set_xlim(0, T_POUR_END)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)

    plt.tight_layout()
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    print(f"Saved: {base}.pdf")
    plt.close()


# ======================================================================
#  PLOT 4 -- position (xyz) and velocity (dxyz) timeseries
#   Two-row panel:
#     Top:    x(t), y(t), z(t)   with obstacle-vicinity shading
#     Bottom: vx(t), vy(t), vz(t), ||v||(t)  with speed limit reference
# ======================================================================
def plot_kinematics(trace, best_cost, base="exp3a_kinematics"):
    pos = trace.position     # (T, 3)
    vel = trace.velocity     # (T, 3)
    t   = trace.time

    speed   = np.linalg.norm(vel, axis=1)
    if OBSTACLE_GEOMETRY == "cylinder_infinite":
        d_obs = np.linalg.norm(pos[:, :2] - OBSTACLE[:2], axis=1)
    else:
        d_obs = np.linalg.norm(pos - OBSTACLE, axis=1)
    d_goal  = np.linalg.norm(pos - GOAL,     axis=1)

    # colour palette — consistent with rest of paper
    c_x   = "#4C72B0"   # blue
    c_y   = "#DD8452"   # orange
    c_z   = "#55A868"   # green
    c_spd = "#8172B2"   # purple for speed magnitude

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # ── Common background: phase bands ──────────────────────────────
    for ax in axes:
        ax.axvspan(0,           T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
        ax.axvspan(T_CARRY_END, T_POUR_END,  alpha=0.025, color=C_POUR,  zorder=0)

        # Obstacle vicinity: shade when d_obs < (OBS_RAD + 0.05)
        obs_near = d_obs < (OBS_RAD + 0.05)
        if np.any(obs_near):
            starts = np.where(np.diff(obs_near.astype(int)) ==  1)[0]
            ends   = np.where(np.diff(obs_near.astype(int)) == -1)[0]
            if obs_near[0]:  starts = np.concatenate([[0], starts])
            if obs_near[-1]: ends   = np.concatenate([ends, [len(obs_near)-1]])
            for i, (s, e) in enumerate(zip(starts, ends)):
                ax.axvspan(t[s], t[e], alpha=0.12, color="#AAAAAA",
                           label="Near obstacle" if i == 0 else None, zorder=0)

        # Human ramp zone: shade when d_goal < HUMAN_RAMP_RAD
        ramp_near = d_goal < HUMAN_RAMP_RAD
        if np.any(ramp_near):
            s_ramp = np.where(np.diff(ramp_near.astype(int)) ==  1)[0]
            e_ramp = np.where(np.diff(ramp_near.astype(int)) == -1)[0]
            if ramp_near[0]:  s_ramp = np.concatenate([[0], s_ramp])
            if ramp_near[-1]: e_ramp = np.concatenate([e_ramp, [len(ramp_near)-1]])
            for i, (s, e) in enumerate(zip(s_ramp, e_ramp)):
                ax.axvspan(t[s], t[e], alpha=0.10, color=C_HUMAN,
                           label="Human ramp zone" if i == 0 else None, zorder=0)

    # ── Top panel: position xyz ──────────────────────────────────────
    ax0 = axes[0]
    ax0.plot(t, pos[:, 0], color=c_x, lw=1.8, label=r"$x(t)$")
    ax0.plot(t, pos[:, 1], color=c_y, lw=1.8, label=r"$y(t)$")
    ax0.plot(t, pos[:, 2], color=c_z, lw=1.8, label=r"$z(t)$")

    # Reference lines: start & goal per axis
    for val, col, ls in [
        (START[0], c_x, ":"), (START[1], c_y, ":"), (START[2], c_z, ":"),
        (GOAL[0],  c_x, "--"), (GOAL[1], c_y, "--"), (GOAL[2], c_z, "--"),
    ]:
        ax0.axhline(val, color=col, lw=0.7, ls=ls, alpha=0.35)

    # Phase labels
    ylo0, yhi0 = pos.min() - 0.03, pos.max() + 0.04
    ax0.set_ylim(ylo0, yhi0)
    for tx, lb, col in [(3.5, "Carry", C_CARRY), (8.5, "Pour", C_POUR)]:
        ax0.text(tx, yhi0 - 0.005, lb, fontsize=8, color=col,
                 ha="center", fontweight="bold", alpha=0.65, va="top")

    ax0.set_ylabel("Position (m)", fontsize=11)
    ax0.grid(True, alpha=0.25)
    # Legend: pos axes + shading
    handles0, labels0 = ax0.get_legend_handles_labels()
    ax0.legend(handles0, labels0, fontsize=8.5, loc="upper right",
               framealpha=0.9, edgecolor="lightgrey", fancybox=False, ncol=2)

    # ── Bottom panel: velocity xyz + speed ──────────────────────────
    ax1 = axes[1]
    ax1.plot(t, vel[:, 0], color=c_x, lw=1.6, alpha=0.85, label=r"$\dot{x}(t)$")
    ax1.plot(t, vel[:, 1], color=c_y, lw=1.6, alpha=0.85, label=r"$\dot{y}(t)$")
    ax1.plot(t, vel[:, 2], color=c_z, lw=1.6, alpha=0.85, label=r"$\dot{z}(t)$")
    ax1.plot(t, speed,     color=c_spd, lw=2.2, label=r"$\|\dot{\mathbf{p}}\|$  speed")

    # Speed limit reference
    ax1.axhline( 0.8, color="#333333", ls=":", lw=1.2, alpha=0.65,
                label=f"$v_{{\\rm max}}$ = 0.8 m/s")
    ax1.axhline(-0.8, color="#333333", ls=":", lw=1.2, alpha=0.65)
    ax1.axhline(0.0,  color="#999999", ls="-", lw=0.5, alpha=0.30)

    vhi = max(speed.max(), 0.85) * 1.15
    ax1.set_ylim(-vhi, vhi)

    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Velocity (m/s)", fontsize=11)
    ax1.set_xlim(0, T_POUR_END)
    ax1.grid(True, alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, fontsize=8.5, loc="upper right",
               framealpha=0.9, edgecolor="lightgrey", fancybox=False, ncol=3)

    # ── Shared title ─────────────────────────────────────────────────
    fig.suptitle("Exp 3a — Per-axis Position & Velocity", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    print(f"Saved: {base}.pdf")
    plt.close()


# ======================================================================
#  CSV export — 25 columns in fixed order:
#  x,y,z,qw,qx,qy,qz,
#  k11,k12,k13,k21,k22,k23,k31,k32,k33,
#  d11,d12,d13,d21,d22,d23,d31,d32,d33
# ======================================================================
def save_trajectory_csv(trace, csv_path="exp3a_trajectory.csv"):
    import csv
    pos = trace.position   # (T, 3)
    T = len(trace.time)
    if trace.orientation is not None and len(trace.orientation) == T:
        quat = trace.orientation
    else:
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
    K_arr = trace.gains["K"]   # list of (3,3)
    D_arr = trace.gains["D"]   # list of (3,3)

    header = [
        "x", "y", "z",
        "qw", "qx", "qy", "qz",
        "k11", "k12", "k13",
        "k21", "k22", "k23",
        "k31", "k32", "k33",
        "d11", "d12", "d13",
        "d21", "d22", "d23",
        "d31", "d32", "d33",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(T):
            K = K_arr[i]   # (3,3)
            D = D_arr[i]   # (3,3)
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
    taskspec = load_taskspec_from_json("spec/exp3a_task.json")
    assert taskspec.phases is not None

    global START, GOAL, OBSTACLE, OBS_RAD, OBSTACLE_GEOMETRY
    global Q_UPRIGHT, Q_POUR, T_CARRY_END, T_POUR_END, HUMAN_POS

    START = np.asarray(taskspec.phases[0]["start"], dtype=float)
    GOAL = np.asarray(taskspec.phases[-1]["end"], dtype=float)
    phase_durations = [float(p["duration"]) for p in taskspec.phases]
    T_CARRY_END = phase_durations[0]
    T_POUR_END = float(np.sum(phase_durations))

    Q_UPRIGHT = quat_normalize(np.asarray(taskspec.phases[0].get("start_quat", [1.0, 0.0, 0.0, 0.0]), dtype=float))
    Q_POUR = quat_normalize(np.asarray(taskspec.phases[-1].get("end_quat", [1.0, 0.0, 0.0, 0.0]), dtype=float))

    obs_clause = next((cl for cl in taskspec.clauses if cl.predicate == "ObstacleAvoidance"), None)
    if obs_clause is None:
        raise ValueError("Exp3a requires an ObstacleAvoidance clause in spec/exp3a_task.json")
    OBSTACLE = np.asarray(obs_clause.parameters["obstacle_position"], dtype=float)
    OBS_RAD = float(obs_clause.parameters["safe_radius"])
    OBSTACLE_GEOMETRY = str(obs_clause.parameters.get("geometry", "sphere"))

    ori_limit_clause = next((cl for cl in taskspec.clauses if cl.predicate == "OrientationLimit"), None)
    if ori_limit_clause is not None and "q_ref" in ori_limit_clause.parameters:
        Q_UPRIGHT = quat_normalize(np.asarray(ori_limit_clause.parameters["q_ref"], dtype=float))

    ori_target_clause = next((cl for cl in taskspec.clauses if cl.predicate == "OrientationAtTarget"), None)
    if ori_target_clause is not None and "q_target" in ori_target_clause.parameters:
        Q_POUR = quat_normalize(np.asarray(ori_target_clause.parameters["q_target"], dtype=float))

    HUMAN_POS = GOAL.copy()

    policy    = MultiPhaseCertifiedPolicy(taskspec.phases, K0=300.0, D0=30.0)


    # ── Hard obstacle avoidance by construction ────────────────────────
    # Register hard obstacle from JSON — two-layer avoidance:
    #   Layer 1 (DMP repulsion, inside ODE): steers the spring-damper attractor
    # Layers 1+2 (DMP repulsion + radial projector) are wired automatically
    # from the JSON modality="HARD" clause — no manual hardcoding needed.
    #   Layer 1 (DMP repulsion, inside ODE): repulsive forcing steers trajectory
    #     organically around the obstacle → smooth arc, no C-turn.
    #   Layer 2 (hard radial projector, post-rollout): backstop guarantee.
    #     ∀t: ||p(t)-c|| ≥ r_safe  — by construction, always on.
    policy.setup_hard_obstacles_from_taskspec(taskspec)
    theta_dim = policy.parameter_dimension()
    print(f"Multi-phase policy: {len(taskspec.phases)} phases, theta_dim={theta_dim}")
    print(f"  obstacle geometry: {OBSTACLE_GEOMETRY}, radius={OBS_RAD:.3f} m")
    print(f"  has_orientation: {policy.has_orientation}")
    for i, p in enumerate(taskspec.phases):
        od = policy.ori_dims[i]
        print(f"  Phase {i+1} ({p['label']}): pos_dim={policy.theta_dims[i]-od}, ori_dim={od}")

    predicate_registry = build_predicate_registry()

    compiler     = Compiler(predicate_registry,
                            human_position=HUMAN_POS,
                            human_proximity_radius=HUMAN_PROX_RAD)

    # ── Two-layer obstacle avoidance ──────────────────────────────────
    # Layer 1 (soft, inside PIBB loop):
    #   ObstacleAvoidance clause in scene3_task.json (weight=6, modality=REQUIRE)
    #   nudges the DMP weights toward paths that go around the obstacle.
    #   The optimizer sees a smooth cost signal and shapes the trajectory.
    #
    # Layer 2 (hard, inside every policy.rollout() call):
    #   ObstacleProjector in MultiPhaseCertifiedPolicy.rollout() radially
    #   projects every position point outside the hard boundary BEFORE the Trace
    #   is built.  This runs unconditionally — hard backstop guarantee.
    #
    # Three-layer obstacle avoidance:
    #   Layer 1 — DMP repulsion (inside ODE): smoothly bends the spring-damper
    #     attractor around the obstacle.  Reduces/eliminates C-turn arc.
    #   Layer 2 — Soft cost (ObstacleAvoidance clause in JSON): PIBB reward signal.
    #   Layer 3 — Radial projector (post-rollout): hard guarantee ∀t: ||p-c||≥r.
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
    # dampen pour phase position noise (hold position while rotating)
    # Pour is phase index 1 (0=carry, 1=pour)
    pour_off = policy.offsets[1]
    pour_dim = policy.theta_dims[1]
    sigma_init[pour_off:pour_off + pour_dim] *= 0.05

    optimizer = PIBB(theta=theta_init, sigma=sigma_init, beta=8.0, decay=0.99)

    N_SAMPLES = 30
    N_UPDATES = 70
    best_cost  = float("inf")
    best_theta = theta_init.copy()

    hard_strength = 0.1
    hard_infl = 4
    if obs_clause.hard_obstacle is not None:
        hard_strength = float(obs_clause.hard_obstacle.get("strength", hard_strength))
        hard_infl = float(obs_clause.hard_obstacle.get("infl_factor", hard_infl))

    print(f"\nPIBB: {N_UPDATES} updates x {N_SAMPLES} samples")
    print(f"  K penalty  : ramp [d < {HUMAN_RAMP_RAD} m], hard limit {K_AXIS_LIMIT:.0f} N/m at [d < {HUMAN_PROX_RAD} m]")
    print(
        f"  Obs avoid  : DMP repulsion (str={hard_strength:.3f}, infl={hard_infl:.2f}) "
        f"+ soft cost + hard projector ({OBSTACLE_GEOMETRY}, r={OBS_RAD:.2f} m)"
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
    save_trajectory_csv(trace_final, csv_path="exp3a_trajectory.csv")
    print_diagnostics(trace_final, best_cost)

    plot_3d_workspace(trace_final, best_cost)
    plot_2d_topdown(trace_final, best_cost)
    plot_stiffness(trace_final, best_cost)
    plot_orientation_euler(trace_final, best_cost)
    plot_kinematics(trace_final, best_cost)

    print("Scene 3 done -- 5 plots saved as PNG.")


if __name__ == "__main__":
    main()
