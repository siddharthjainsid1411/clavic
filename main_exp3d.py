"""
Exp 3c: Ball Delivery to Human - HARD Obstacle Avoidance (no pour).

Same geometry as Scene 3b, but obstacle avoidance is HARD:
post-rollout radial projection + localized Gaussian deformation
(geometric guarantee).

2-phase task:
  Phase 1 (0-7s) : carry ball from start to human
  Phase 2 (7-10s): hold at human position (no pour)

CGMS guarantee K=Q^T Q > 0 is maintained throughout.
Human-proximity stiffness reduction is active (same compiler cost).

Plots (PNG only):
    1. exp3c_workspace.png      - 3D Franka workspace view
    2. exp3c_topdown.png        - 2D X-Y top-down view
    3. exp3c_stiffness.png      - Per-axis Kxx/Kyy/Kzz vs time
    4. exp3c_orientation.png    - quaternion tracking + angular error to q_ref
    5. exp3c_kinematics.png     - Position and velocity timeseries
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

# -- seaborn style ------------------------------------------------------
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# -- scene values (loaded from JSON in main) ---------------------------
START = None
GOAL = None
OBSTACLE = None
OBS_RAD = None
OBS_SAFE_RAD = None
OBSTACLE_GEOMETRY = "sphere"

Q_UPRIGHT = None

# 2-phase timing: carry + hold
T_CARRY_END = None
T_HOLD_END  = None

HUMAN_POS      = None
HUMAN_PROX_RAD = 0.12
HUMAN_RAMP_RAD = 0.36     # ramp starts at 3x proximity radius (matches compiler RAMP_FACTOR=3)
K_AXIS_LIMIT   = 100.0    # N/m per axis - target inside proximity radius

# obstacle box half-extents for plot (scaled from circular radius)
OBS_BOX_HX_SCALE = 0.844  # measured laptop footprint ratio: hx / r
OBS_BOX_HY_SCALE = 0.547  # measured laptop footprint ratio: hy / r
OBS_HX, OBS_HY, OBS_HZ = None, None, 0.05

# -- colours ------------------------------------------------------------
C_CARRY  = "#4C72B0"
C_HOLD   = "#55A868"    # green for hold
C_OBS    = "#AAAAAA"
C_HUMAN  = "#AEC7E8"
C_START  = "#2CA02C"
C_GOAL   = "#1F77B4"
C_DASH   = "#999999"
C_KX     = "#4C72B0"
C_KY     = "#DD8452"
C_KZ     = "#55A868"
C_BALL   = "#FF7F0E"    # orange accent for ball label

# -- annotation ---------------------------------------------------------
SCENE_LABEL = "Exp 3c - Ball Delivery (HARD obstacle avoidance, no pour)"


# -- predicates ---------------------------------------------------------
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


# -- quaternion -> Euler ZYX (degrees) ----------------------------------
def quat_to_euler(q):
    """Returns (roll_x, pitch_y, yaw_z) in degrees."""
    w, x, y, z = q
    roll  = np.degrees(np.arctan2(2*(w*x + y*z),  1 - 2*(x*x + y*y)))
    sinp  = np.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y),  1 - 2*(y*y + z*z)))
    return roll, pitch, yaw


def smooth_quaternion_signs(q_series):
    """Ensure sign continuity (q and -q represent same rotation)."""
    q_cont = np.asarray(q_series, dtype=float).copy()
    if len(q_cont) == 0:
        return q_cont

    q_cont[0] = quat_normalize(q_cont[0])
    for i in range(1, len(q_cont)):
        qi = quat_normalize(q_cont[i])
        if np.dot(qi, q_cont[i - 1]) < 0.0:
            qi = -qi
        q_cont[i] = qi
    return q_cont


def _distance_to_obstacle(positions):
    positions = np.asarray(positions, dtype=float)
    if OBSTACLE_GEOMETRY == "cylinder_infinite":
        return np.linalg.norm(positions[:, :2] - OBSTACLE[:2], axis=1)
    return np.linalg.norm(positions - OBSTACLE, axis=1)


# -- diagnostics --------------------------------------------------------
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

    d_obs     = _distance_to_obstacle(pos)
    obs_cm    = (d_obs.min() - OBS_RAD) * 100.0
    n_inside  = int(np.sum(d_obs < OBS_SAFE_RAD))

    K_eig_min = float(min(np.linalg.eigvalsh(K)[0] for K in K_arr))

    near_mask = d_goal < HUMAN_PROX_RAD
    ramp_mask = d_goal < HUMAN_RAMP_RAD

    hold_mask  = trace.time > T_CARRY_END
    hold_drift = float(np.linalg.norm(pos[hold_mask] - GOAL, axis=1).max()) \
                 if np.any(hold_mask) else 0.0

    sep = "=" * 48
    print(f"\n{sep} EXP 3c DIAGNOSTICS {sep}")
    print("  Scene             : Ball delivery (HARD obstacle avoidance, no pour)")
    print(f"  Best cost         : {best_cost:.4f}")
    print(f"  Goal reached      : {'YES' if reached else 'NO'}  t={t_reach:.2f} s")
    print(f"  Max speed         : {speed.max():.4f} m/s  (limit 0.8)")
    print(f"  Obstacle clearance: {obs_cm:.1f} cm  (HARD {OBSTACLE_GEOMETRY}, r={OBS_RAD:.2f} m)")
    print(f"  Pts inside obs    : {n_inside}  (HARD: must be 0)")
    print(f"  Hold pos drift    : {hold_drift*100:.1f} cm  (target < 5)")
    print(f"  tr(K) range       : [{trK.min():.0f}, {trK.max():.0f}] N/m")
    print(f"  tr(D) range       : [{trD.min():.1f}, {trD.max():.1f}] Ns/m")
    print(f"  K eigenvalue min  : {K_eig_min:.4f}  (CGMS > 0 required)")

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
    print(f"{sep * 2}\n")


# -- plot helpers -------------------------------------------------------
def to_plot(p):
    """Identity mapping - world [x,y,z] -> plot [x,y,z]."""
    return np.asarray(p)


def _obs_box_faces(cx, cy, cz, dx, dy, dz_bot, dz_top):
    """Return 6 face lists for a box layer from cz+dz_bot to cz+dz_top."""
    corners = np.array([
        [cx-dx, cy-dy, cz+dz_bot], [cx+dx, cy-dy, cz+dz_bot],
        [cx+dx, cy+dy, cz+dz_bot], [cx-dx, cy+dy, cz+dz_bot],
        [cx-dx, cy-dy, cz+dz_top], [cx+dx, cy-dy, cz+dz_top],
        [cx+dx, cy+dy, cz+dz_top], [cx-dx, cy+dy, cz+dz_top],
    ])
    return [
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[0], corners[3], corners[7], corners[4]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[2], corners[3]],
    ]


# ======================================================================
#  PLOT 1 -- 3D workspace
# ======================================================================
def plot_3d_workspace(trace, best_cost, base="exp3c_workspace"):
    pos = trace.position
    t   = trace.time

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # -- obstacle cuboid: bottom 20% solid base, top 80% translucent ------
    cx, cy, cz = OBSTACLE
    dx, dy, dz = OBS_HX, OBS_HY, OBS_HZ
    split_z = -dz + 2 * dz * 0.20
    ax.add_collection3d(Poly3DCollection(
        _obs_box_faces(cx, cy, cz, dx, dy, -dz, split_z),
        alpha=0.90, facecolor="#1A1A1A", edgecolor="#000000", linewidth=0.7
    ))
    ax.add_collection3d(Poly3DCollection(
        _obs_box_faces(cx, cy, cz, dx, dy, split_z, +dz),
        alpha=0.20, facecolor=C_OBS, edgecolor="#666666", linewidth=0.5
    ))

    # human proximity cylinder
    r_cyl   = HUMAN_PROX_RAD * 0.6
    h_cyl   = 0.35
    gx, gy, gz = to_plot(GOAL)
    z_bot = gz - 0.05
    z_top = z_bot + h_cyl
    theta_c  = np.linspace(0, 2*np.pi, 40)
    xc = gx + r_cyl * np.cos(theta_c)
    yc = gy + r_cyl * np.sin(theta_c)
    for zb in [z_bot, z_top]:
        ax.plot(xc, yc, [zb]*len(theta_c), color=C_HUMAN, lw=0.6, alpha=0.45)
    for i in range(0, len(theta_c), 4):
        ax.plot([xc[i], xc[i]], [yc[i], yc[i]], [z_bot, z_top],
                color=C_HUMAN, lw=0.5, alpha=0.30)
    z_fill    = np.linspace(z_bot, z_top, 2)
    Theta, Zg = np.meshgrid(theta_c, z_fill)
    ax.plot_surface(gx + r_cyl*np.cos(Theta), gy + r_cyl*np.sin(Theta), Zg,
                    alpha=0.10, color=C_HUMAN, edgecolor="none")

    # straight path
    sp = to_plot(START); gp = to_plot(GOAL)
    ax.plot([sp[0], gp[0]], [sp[1], gp[1]], [sp[2], gp[2]],
            "--", color=C_DASH, lw=1.8, alpha=0.60, label="Shortest path", zorder=2)

    # trajectory coloured by phase
    ip = np.searchsorted(t, T_CARRY_END)
    pp = to_plot(pos)
    ax.plot(pp[:ip+1, 0], pp[:ip+1, 1], pp[:ip+1, 2],
            color=C_CARRY, lw=2.2, solid_capstyle="round", zorder=5, label="Carry")
    ax.plot(pp[ip:, 0], pp[ip:, 1], pp[ip:, 2],
            color=C_HOLD,  lw=2.2, solid_capstyle="round", zorder=5, label="Hold")

    # markers
    ax.scatter(sp[0], sp[1], sp[2], s=65, c=C_START, zorder=10,
               depthshade=False, edgecolors="black", linewidth=0.6)
    ax.scatter(gp[0], gp[1], gp[2], s=60, c=C_GOAL,  zorder=10,
               depthshade=False, edgecolors="black", linewidth=0.6, marker="D")
    ax.text(sp[0]+0.02, sp[1]-0.01, sp[2]+0.03, "Start",
            fontsize=8, fontweight="bold", color=C_START)
    ax.text(gp[0]+0.03, gp[1]+0.01, gp[2]+0.03, "Human\n(goal)",
            fontsize=8, fontweight="bold", color=C_GOAL)
    obs_p = to_plot(OBSTACLE)
    ax.text(obs_p[0], obs_p[1]+0.04, obs_p[2]+0.03, "Obstacle\n(hard)",
            fontsize=7, color="#555555", ha="center")

    ax.set_xlabel("X - forward/depth (m)", fontsize=9, labelpad=7)
    ax.set_ylabel("Y - lateral (m)",        fontsize=9, labelpad=7)
    ax.set_zlabel("Z - height (m)",         fontsize=9, labelpad=7)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=20, azim=-55)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("lightgrey")
    ax.grid(True, alpha=0.28)

    extra = [
        mpatches.Patch(facecolor="#1A1A1A", alpha=0.90, edgecolor="#000",
                       label="Obstacle base (solid)"),
        mpatches.Patch(facecolor=C_OBS,     alpha=0.30, edgecolor="#666",
                       label="Obstacle upper (hard avoidance)"),
        mpatches.Patch(facecolor=C_HUMAN,   alpha=0.25, edgecolor=C_HUMAN,
                       label=f"Human zone (r={r_cyl:.2f} m)"),
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
#  PLOT 2 -- 2D top-down view
# ======================================================================
def plot_2d_topdown(trace, best_cost, base="exp3c_topdown"):
    pos = trace.position
    t   = trace.time

    fig, ax = plt.subplots(figsize=(8.2, 6.8))

    # Plot laptop as square footprint for visualization.
    side_half = max(OBS_HX, OBS_HY)
    obs_rect = mpatches.Rectangle(
        (OBSTACLE[0] - side_half, OBSTACLE[1] - side_half),
        2 * side_half,
        2 * side_half,
        facecolor=C_OBS,
        edgecolor="#777777",
        linewidth=1.0,
        alpha=0.25,
        zorder=2,
        label="Laptop footprint (square plot)",
    )
    ax.add_patch(obs_rect)
    obs_safe = plt.Circle((OBSTACLE[0], OBSTACLE[1]), OBS_RAD,
                          fill=False, color="#777777", linestyle="--",
                          linewidth=1.2, alpha=0.85, zorder=3,
                          label=f"Obstacle model circle (r={OBS_RAD:.2f} m)")
    ax.add_patch(obs_safe)

    ax.text(OBSTACLE[0], OBSTACLE[1] + side_half + 0.025, "Obstacle\n(square plot, circle cost)",
            fontsize=7.5, ha="center", color="#555555", style="italic")

    # human proximity zones
    ramp_circle = plt.Circle((GOAL[0], GOAL[1]), HUMAN_RAMP_RAD,
                              color=C_HUMAN, alpha=0.08, zorder=1,
                              label=f"Ramp zone (r={HUMAN_RAMP_RAD:.2f} m)")
    hard_circle = plt.Circle((GOAL[0], GOAL[1]), HUMAN_PROX_RAD,
                              color=C_HUMAN, alpha=0.20, zorder=1,
                              label=f"Hard zone (r={HUMAN_PROX_RAD:.2f} m)")
    ax.add_patch(ramp_circle)
    ax.add_patch(hard_circle)

    # straight-line path
    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]],
            "--", color=C_DASH, lw=1.6, alpha=0.55, zorder=3, label="Shortest path")

    # trajectory coloured by phase
    ip = np.searchsorted(t, T_CARRY_END)
    ax.plot(pos[:ip+1, 0], pos[:ip+1, 1], color=C_CARRY, lw=2.2,
            solid_capstyle="round", zorder=5, label="Carry (ball)")
    ax.plot(pos[ip:, 0],   pos[ip:, 1],   color=C_HOLD,  lw=2.2,
            solid_capstyle="round", zorder=5, label="Hold")

    # start / goal markers
    ax.scatter(START[0], START[1], s=80, c=C_START, zorder=10,
               edgecolors="black", linewidth=0.7, label="Start")
    ax.scatter(GOAL[0],  GOAL[1],  s=70, c=C_GOAL,  zorder=10,
               edgecolors="black", linewidth=0.7, marker="D", label="Human (goal)")
    ax.text(START[0] + 0.01, START[1] - 0.035, "Start",
            fontsize=8, fontweight="bold", color=C_START)
    ax.text(GOAL[0]  + 0.01, GOAL[1]  + 0.015, "Human\n(goal)",
            fontsize=8, fontweight="bold", color=C_GOAL)

    ax.set_xlabel("X - forward/depth (m)", fontsize=11)
    ax.set_ylabel("Y - lateral (m)",        fontsize=11)
    ax.set_title(f"Top-down view: table plane (X-Y)\n{SCENE_LABEL}", fontsize=10)
    ax.set_aspect("equal")

    x_candidates = [
        START[0], GOAL[0],
        OBSTACLE[0] - max(OBS_RAD, side_half), OBSTACLE[0] + max(OBS_RAD, side_half),
        GOAL[0] - HUMAN_RAMP_RAD, GOAL[0] + HUMAN_RAMP_RAD,
    ]
    y_candidates = [
        START[1], GOAL[1],
        OBSTACLE[1] - max(OBS_RAD, side_half), OBSTACLE[1] + max(OBS_RAD, side_half),
        GOAL[1] - HUMAN_RAMP_RAD, GOAL[1] + HUMAN_RAMP_RAD,
    ]
    margin = 0.04
    xlo = min(x_candidates) - margin
    xhi = max(x_candidates) + margin
    ylo = min(y_candidates) - margin
    yhi = max(y_candidates) + margin
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)

    plt.tight_layout(rect=[0.0, 0.0, 0.79, 1.0])
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ======================================================================
#  PLOT 3 -- per-axis stiffness vs time
# ======================================================================
def plot_stiffness(trace, best_cost, base="exp3c_stiffness"):
    K_arr  = trace.gains["K"]
    Kd     = np.array([[K[0,0], K[1,1], K[2,2]] for K in K_arr])
    t      = trace.time
    d_goal = np.linalg.norm(trace.position - GOAL, axis=1)

    ramp_mask = d_goal < HUMAN_RAMP_RAD
    hard_mask = d_goal < HUMAN_PROX_RAD

    fig, ax = plt.subplots(figsize=(9, 4.2))

    ax.axvspan(0,           T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
    ax.axvspan(T_CARRY_END, T_HOLD_END,  alpha=0.025, color=C_HOLD,  zorder=0)

    if np.any(ramp_mask):
        starts = np.where(np.diff(ramp_mask.astype(int)) ==  1)[0]
        ends   = np.where(np.diff(ramp_mask.astype(int)) == -1)[0]
        if ramp_mask[0]:  starts = np.concatenate([[0], starts])
        if ramp_mask[-1]: ends   = np.concatenate([ends, [len(ramp_mask)-1]])
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(t[s], t[e], alpha=0.08, color=C_HUMAN,
                       label="Ramp zone (penalty onset)" if i == 0 else None, zorder=0)

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
               label=fr"$K_{{\rm limit}}$ = {K_AXIS_LIMIT:.0f} N/m")

    ax.set_ylim(bottom=0)
    yhi = max(Kd.max() * 1.10, K_AXIS_LIMIT * 1.5)
    ax.set_ylim(0, yhi)
    for tx, lb, col in [(3.5, "Carry", C_CARRY), (8.5, "Hold", C_HOLD)]:
        ax.text(tx, yhi * 0.94, lb, fontsize=8, color=col,
                ha="center", fontweight="bold", alpha=0.70)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Stiffness (N/m)", fontsize=11)
    ax.set_title(SCENE_LABEL, fontsize=9)
    ax.set_xlim(0, T_HOLD_END)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False, ncol=2)

    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ======================================================================
#  PLOT 4 -- Euler orientation vs time
# ======================================================================
def plot_orientation_euler(trace, best_cost, base="exp3c_orientation"):
    if trace.orientation is None:
        print("No orientation - skipping orientation plot.")
        return
    q = smooth_quaternion_signs(trace.orientation)
    t = trace.time
    q_ref = quat_normalize(Q_UPRIGHT)
    ang_err = np.degrees(np.array([quat_distance(quat_normalize(qi), q_ref) for qi in q]))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)

    for ax in (ax0, ax1):
        ax.axvspan(0,           T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
        ax.axvspan(T_CARRY_END, T_HOLD_END,  alpha=0.025, color=C_HOLD,  zorder=0)
        ax.grid(True, alpha=0.25)

    ax0.plot(t, q[:, 0], color="#4C72B0", lw=2.0, label="qw")
    ax0.plot(t, q[:, 1], color="#DD8452", lw=2.0, label="qx")
    ax0.plot(t, q[:, 2], color="#55A868", lw=2.0, label="qy")
    ax0.plot(t, q[:, 3], color="#C44E52", lw=2.0, label="qz")
    ax0.axhline(q_ref[0], color="#4C72B0", ls=":", lw=1.0, alpha=0.50)
    ax0.axhline(q_ref[1], color="#DD8452", ls=":", lw=1.0, alpha=0.50)
    ax0.axhline(q_ref[2], color="#55A868", ls=":", lw=1.0, alpha=0.50)
    ax0.axhline(q_ref[3], color="#C44E52", ls=":", lw=1.0, alpha=0.50)
    ax0.set_ylabel("Quat comp.", fontsize=10)
    ax0.set_ylim(-1.05, 1.05)
    ax0.legend(fontsize=8.5, loc="upper right", framealpha=0.9,
               edgecolor="lightgrey", fancybox=False, ncol=2)
    ax0.set_title(f"{SCENE_LABEL}\nOrientation tracking", fontsize=10)

    ax1.plot(t, ang_err, color="#8172B2", lw=2.2,
             label="Angular error to q_ref")
    ax1.axhline(0.0, color="#999999", ls="-", lw=0.7, alpha=0.40)
    ax1.set_ylabel("Error (deg)", fontsize=10)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_xlim(0, T_HOLD_END)
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.9,
               edgecolor="lightgrey", fancybox=False)

    y0 = max(5.0, float(np.max(ang_err) * 1.15))
    ax1.set_ylim(0.0, y0)

    for tx, lb, col in [(0.5 * T_CARRY_END, "Carry", C_CARRY),
                        (0.5 * (T_CARRY_END + T_HOLD_END), "Hold", C_HOLD)]:
        ax1.text(tx, y0 * 0.90, lb, fontsize=8, color=col,
                 ha="center", fontweight="bold", alpha=0.70)

    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ======================================================================
#  PLOT 5 -- position and velocity timeseries
# ======================================================================
def plot_kinematics(trace, best_cost, base="exp3c_kinematics"):
    pos   = trace.position
    vel   = trace.velocity
    t     = trace.time
    speed = np.linalg.norm(vel, axis=1)
    d_obs = _distance_to_obstacle(pos)
    d_goal= np.linalg.norm(pos - GOAL,     axis=1)

    c_x = "#4C72B0"; c_y = "#DD8452"; c_z = "#55A868"; c_spd = "#8172B2"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for ax in axes:
        ax.axvspan(0,           T_CARRY_END, alpha=0.025, color=C_CARRY, zorder=0)
        ax.axvspan(T_CARRY_END, T_HOLD_END,  alpha=0.025, color=C_HOLD,  zorder=0)

        obs_near = d_obs < (OBS_RAD + 0.05)
        if np.any(obs_near):
            starts = np.where(np.diff(obs_near.astype(int)) ==  1)[0]
            ends   = np.where(np.diff(obs_near.astype(int)) == -1)[0]
            if obs_near[0]:  starts = np.concatenate([[0], starts])
            if obs_near[-1]: ends   = np.concatenate([ends, [len(obs_near)-1]])
            for i, (s, e) in enumerate(zip(starts, ends)):
                ax.axvspan(t[s], t[e], alpha=0.10, color="#AAAAAA",
                           label="Near obstacle" if i == 0 else None, zorder=0)

        ramp_near = d_goal < HUMAN_RAMP_RAD
        if np.any(ramp_near):
            s_ramp = np.where(np.diff(ramp_near.astype(int)) ==  1)[0]
            e_ramp = np.where(np.diff(ramp_near.astype(int)) == -1)[0]
            if ramp_near[0]:  s_ramp = np.concatenate([[0], s_ramp])
            if ramp_near[-1]: e_ramp = np.concatenate([e_ramp, [len(ramp_near)-1]])
            for i, (s, e) in enumerate(zip(s_ramp, e_ramp)):
                ax.axvspan(t[s], t[e], alpha=0.10, color=C_HUMAN,
                           label="Human ramp zone" if i == 0 else None, zorder=0)

    ax0 = axes[0]
    ax0.plot(t, pos[:, 0], color=c_x, lw=1.8, label=r"$x(t)$")
    ax0.plot(t, pos[:, 1], color=c_y, lw=1.8, label=r"$y(t)$")
    ax0.plot(t, pos[:, 2], color=c_z, lw=1.8, label=r"$z(t)$")
    for val, col, ls in [
        (START[0], c_x, ":"), (START[1], c_y, ":"), (START[2], c_z, ":"),
        (GOAL[0],  c_x, "--"), (GOAL[1], c_y, "--"), (GOAL[2], c_z, "--"),
    ]:
        ax0.axhline(val, color=col, lw=0.7, ls=ls, alpha=0.35)
    ylo0, yhi0 = pos.min() - 0.03, pos.max() + 0.04
    ax0.set_ylim(ylo0, yhi0)
    for tx, lb, col in [(3.5, "Carry", C_CARRY), (8.5, "Hold", C_HOLD)]:
        ax0.text(tx, yhi0 - 0.005, lb, fontsize=8, color=col,
                 ha="center", fontweight="bold", alpha=0.65, va="top")
    ax0.set_ylabel("Position (m)", fontsize=11)
    ax0.grid(True, alpha=0.25)
    handles0, labels0 = ax0.get_legend_handles_labels()
    ax0.legend(handles0, labels0, fontsize=8.5, loc="upper right",
               framealpha=0.9, edgecolor="lightgrey", fancybox=False, ncol=2)

    ax1 = axes[1]
    ax1.plot(t, vel[:, 0], color=c_x, lw=1.6, alpha=0.85, label=r"$\dot{x}(t)$")
    ax1.plot(t, vel[:, 1], color=c_y, lw=1.6, alpha=0.85, label=r"$\dot{y}(t)$")
    ax1.plot(t, vel[:, 2], color=c_z, lw=1.6, alpha=0.85, label=r"$\dot{z}(t)$")
    ax1.plot(t, speed,     color=c_spd, lw=2.2, label=r"$\|\dot{\mathbf{p}}\|$  speed")
    ax1.axhline( 0.8, color="#333333", ls=":", lw=1.2, alpha=0.65,
                label=f"$v_{{\rm max}}$ = 0.8 m/s")
    ax1.axhline(-0.8, color="#333333", ls=":", lw=1.2, alpha=0.65)
    ax1.axhline(0.0,  color="#999999", ls="-", lw=0.5, alpha=0.30)
    vhi = max(speed.max(), 0.85) * 1.15
    ax1.set_ylim(-vhi, vhi)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Velocity (m/s)", fontsize=11)
    ax1.set_xlim(0, T_HOLD_END)
    ax1.grid(True, alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, fontsize=8.5, loc="upper right",
               framealpha=0.9, edgecolor="lightgrey", fancybox=False, ncol=3)

    fig.suptitle("Exp 3c - Per-axis Position and Velocity", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()



def save_trajectory_csv(trace, csv_path="exp3c_trajectory.csv"):
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
    global START, GOAL, OBSTACLE, OBS_RAD, OBS_SAFE_RAD, OBSTACLE_GEOMETRY
    global Q_UPRIGHT, T_CARRY_END, T_HOLD_END
    global HUMAN_POS, OBS_HX, OBS_HY

    taskspec = load_taskspec_from_json("spec/exp3c_task.json")
    assert taskspec.phases is not None

    phases = taskspec.phases
    START = np.asarray(phases[0]["start"], dtype=float)
    GOAL = np.asarray(phases[-1]["end"], dtype=float)
    durations = [float(p.get("duration", 0.0)) for p in phases]
    T_CARRY_END = durations[0] if durations else 0.0
    T_HOLD_END = float(np.sum(durations)) if durations else 0.0
    Q_UPRIGHT = np.asarray(phases[0].get("start_quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)

    obs_clause = next((c for c in taskspec.clauses
                       if c.predicate == "ObstacleAvoidance"), None)
    if obs_clause is None:
        raise ValueError("exp3c_task.json must include an ObstacleAvoidance clause")

    if "obstacle_position" not in obs_clause.parameters or "safe_radius" not in obs_clause.parameters:
        raise ValueError("ObstacleAvoidance clause must provide obstacle_position and safe_radius")

    OBSTACLE = np.asarray(obs_clause.parameters["obstacle_position"], dtype=float)
    OBS_RAD = float(obs_clause.parameters["safe_radius"])
    OBS_SAFE_RAD = OBS_RAD
    OBSTACLE_GEOMETRY = str(obs_clause.parameters.get("geometry", "sphere"))

    ori_clause = next((c for c in taskspec.clauses
                       if c.predicate == "OrientationLimit"), None)
    if ori_clause is not None and "q_ref" in ori_clause.parameters:
        Q_UPRIGHT = np.asarray(ori_clause.parameters["q_ref"], dtype=float)

    HUMAN_POS = GOAL.copy()
    OBS_HX = max(0.04, OBS_RAD * OBS_BOX_HX_SCALE)
    OBS_HY = max(0.04, OBS_RAD * OBS_BOX_HY_SCALE)

    policy = MultiPhaseCertifiedPolicy(taskspec.phases, K0=300.0, D0=30.0)

    # HARD avoidance: post-rollout geometric deformation from JSON.
    policy.setup_hard_obstacles_from_taskspec(taskspec)

    theta_dim = policy.parameter_dimension()
    print("Exp 3c: Ball delivery - HARD obstacle avoidance (no pour)")
    print("  avoidance=HARD: post-rollout radial projection + localized Gaussian smoothing")
    print(f"Multi-phase policy: {len(taskspec.phases)} phases, theta_dim={theta_dim}")
    print(f"  has_orientation: {policy.has_orientation}")
    for i, p in enumerate(taskspec.phases):
        od = policy.ori_dims[i]
        print(f"  Phase {i+1} ({p['label']}): pos_dim={policy.theta_dims[i]-od}, ori_dim={od}")

    predicate_registry = build_predicate_registry()
    compiler = Compiler(predicate_registry,
                        human_position=HUMAN_POS,
                        human_proximity_radius=HUMAN_PROX_RAD)
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
    hold_off = policy.offsets[1]
    hold_dim = policy.theta_dims[1]
    sigma_init[hold_off:hold_off + hold_dim] *= 0.05

    optimizer = PIBB(theta=theta_init, sigma=sigma_init, beta=8.0, decay=0.99)

    N_SAMPLES = 30
    N_UPDATES = 70
    best_cost  = float("inf")
    best_theta = theta_init.copy()

    print(f"\nPIBB: {N_UPDATES} updates x {N_SAMPLES} samples")
    print(
        f"  Obstacle: HARD (geometry={OBSTACLE_GEOMETRY}, "
        f"post-rollout radial projection + localized Gaussian smoothing, "
        f"safe r={OBS_SAFE_RAD:.2f} m)"
    )
    print(f"  K penalty: ramp [d < {HUMAN_RAMP_RAD} m], limit {K_AXIS_LIMIT:.0f} N/m at [d < {HUMAN_PROX_RAD} m]")

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
            print(f"  [{upd+1:03d}]  min={costs_s.min():.4f}  "
                  f"mean(clipped)={costs_s.mean():.4f}  best={best_cost:.4f}")

    print("Optimization complete.\n")

    trace_final = policy.rollout(best_theta)
    save_trajectory_csv(trace_final, csv_path="exp3c_trajectory.csv")
    print_diagnostics(trace_final, best_cost)

    plot_3d_workspace(trace_final, best_cost)
    plot_2d_topdown(trace_final, best_cost)
    plot_stiffness(trace_final, best_cost)
    plot_orientation_euler(trace_final, best_cost)
    plot_kinematics(trace_final, best_cost)

    print("Scene 3c done - 5 plots saved as PNG (no PDF).")


if __name__ == "__main__":
    main()
