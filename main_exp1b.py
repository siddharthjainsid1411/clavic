"""
Exp 1b: Carry to goal in 2 s — Human standing in workspace (comfort violation demo).

Demonstrates HARD vs SOFT constraint behaviour under time pressure:
  Body exclusion  (REQUIRE + HARD DMP + projector, r=0.08 m):
      Geometric guarantee — ALWAYS respected, regardless of time budget.
  Comfort zone    (PREFER, weight=4.0 — reduced, r=0.19 m):
      Soft cost — optimizer trades comfort violation against reaching goal
      within 4 s.  Comfort zone WILL likely be entered.

Compare with Exp 1 (T=10 s, weight=15): trajectory stays fully outside.
Here (T=2 s, weight=4): trajectory cuts through comfort zone — body still safe.

Geometry:
    Start, goal, and human positions are loaded from spec/exp1b_task.json

Outputs (prefix scene5b_):
  scene5b_workspace.png / scene5b_topdown.png
  scene5b_stiffness.png / scene5b_orientation.png / scene5b_kinematics.png
    scene5b_obstacle_hocbf.png
  scene5b_checkpoint.npz / scene5b_trajectory.csv
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import seaborn as sns

from spec.compiler import Compiler
from spec.json_parser import load_taskspec_from_json
from core.multi_phase_policy import MultiPhaseCertifiedPolicy
from optimization.optimizer import PIBB
from core.cgms.quat_utils import quat_normalize

from logic.predicates import (
    at_waypoint,
    human_body_exclusion,
    human_comfort_distance,
    velocity_limit,
    orientation_limit,
    angular_velocity_limit,
)

# ── style ──────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# ── scene constants (loaded from JSON in main()) ─────────────────────
START  = None
GOAL   = None
HUMAN  = None

HUMAN_BODY_RAD    = 0.08
HUMAN_COMFORT_RAD = 0.19
HUMAN_RAMP_RAD    = 3.0 * HUMAN_COMFORT_RAD
HUMAN_GEOMETRY    = "sphere"
K_AXIS_LIMIT      = 100.0

HORIZON = 2.0

# ── colours ────────────────────────────────────────────────────────────
C_CARRY   = "#9467BD"
C_BODY    = "#E74C3C"
C_COMFORT = "#F39C12"
C_START   = "#2CA02C"
C_GOAL    = "#1F77B4"
C_DASH    = "#999999"
C_KX      = "#4C72B0"
C_KY      = "#DD8452"
C_KZ      = "#55A868"

SCENE_LABEL = "Exp 1b — Carry to Goal (T=2 s): HARD body + SOFT comfort (reduced weight)"


# ── predicate registry ────────────────────────────────────────────────
def build_predicate_registry():
    return {
        "AtGoal":                at_waypoint,
        "HumanBodyExclusion":    human_body_exclusion,
        "HumanComfortDistance":  human_comfort_distance,
        "VelocityLimit":         velocity_limit,
        "OrientationLimit":      orientation_limit,
        "AngularVelocityLimit":  angular_velocity_limit,
    }


# ── quaternion → Euler ZYX (degrees) ─────────────────────────────────
def quat_to_euler(q):
    w, x, y, z = q
    roll  = np.degrees(np.arctan2(2*(w*x + y*z),  1 - 2*(x*x + y*y)))
    sinp  = np.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y),  1 - 2*(y*y + z*z)))
    return roll, pitch, yaw


def _distance_to_human(positions):
    positions = np.asarray(positions, dtype=float)
    if HUMAN_GEOMETRY == "cylinder_infinite":
        return np.linalg.norm(positions[:, :2] - HUMAN[:2], axis=1)
    return np.linalg.norm(positions - HUMAN, axis=1)


# ── diagnostics ───────────────────────────────────────────────────────
def print_diagnostics(trace, best_cost):
    pos   = trace.position
    speed = np.linalg.norm(trace.velocity, axis=1)
    K_arr = trace.gains["K"]
    D_arr = trace.gains["D"]
    t     = trace.time

    trK    = np.array([np.trace(K) for K in K_arr])
    trD    = np.array([np.trace(D) for D in D_arr])
    K_diag = np.array([[K[0,0], K[1,1], K[2,2]] for K in K_arr])

    d_human = _distance_to_human(pos)
    d_goal  = np.linalg.norm(pos - GOAL,  axis=1)

    body_cm    = (d_human.min() - HUMAN_BODY_RAD)    * 100.0
    comfort_cm = (d_human.min() - HUMAN_COMFORT_RAD) * 100.0
    n_in_body    = int(np.sum(d_human < HUMAN_BODY_RAD))
    n_in_comfort = int(np.sum(d_human < HUMAN_COMFORT_RAD))

    reached = np.any(d_goal < 0.06)
    t_reach = float(t[np.argmin(d_goal)]) if reached else float("nan")

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

    velocity_cbf = trace.safety.get("velocity_cbf", {}) if hasattr(trace, "safety") else {}
    if velocity_cbf:
        print(f"  Velocity CBF active    : {_mask_intervals(t, velocity_cbf.get('active', []))}")
        if "window_active" in velocity_cbf:
            print(f"  Velocity windows       : {_mask_intervals(t, velocity_cbf.get('window_active', []))}")
    orientation_hocbf = trace.safety.get("orientation_hocbf", {}) if hasattr(trace, "safety") else {}
    if orientation_hocbf:
        print(f"  Orientation HOCBF      : {_mask_intervals(t, orientation_hocbf.get('active', []))}")
    angular_velocity_cbf = trace.safety.get("angular_velocity_cbf", {}) if hasattr(trace, "safety") else {}
    if angular_velocity_cbf:
        print(f"  AngVel CBF active      : {_mask_intervals(t, angular_velocity_cbf.get('active', []))}")

    sep = "=" * 50
    print(f"\n{sep} EXP 1b DIAGNOSTICS {sep}")
    print(f"  Horizon                : {HORIZON} s  (short — comfort violation expected)")
    print(f"  Best cost              : {best_cost:.4f}")
    print(f"  Goal reached           : {'YES' if reached else 'NO'}  t={t_reach:.2f} s")
    print(f"  Max speed              : {speed.max():.4f} m/s  (limit 0.8)")
    print(f"  --- Human Body (HARD {HUMAN_GEOMETRY}, r={HUMAN_BODY_RAD:.2f} m) ---")
    print(f"  Min clearance (body)   : {body_cm:+.1f} cm  (must be > 0)")
    print(f"  Pts inside body        : {n_in_body}  (must be 0)")
    if obstacle_hocbf:
        print(
            f"  Obstacle HOCBF         : active_steps={obs_active_steps}, "
            f"min_h={obs_h_min}, min_hdot={obs_hdot_min}"
        )
        print(f"  Obstacle HOCBF on      : {_mask_intervals(t, obstacle_hocbf.get('active', []))}")
    if proj:
        print(f"  Projection active      : steps={proj_active_steps}")
        print(f"  Projection windows     : {_mask_intervals(t, proj.get('active', []))}")
    print(f"  --- Comfort Zone (SOFT w=4, r={HUMAN_COMFORT_RAD:.2f} m) ---")
    print(f"  Min clearance (comfort): {comfort_cm:+.1f} cm  (negative = violated, expected here)")
    print(f"  Pts inside comfort     : {n_in_comfort}  (soft — violation allowed)")
    print(f"  --- Stiffness ---")
    print(f"  tr(K) range            : [{trK.min():.0f}, {trK.max():.0f}] N/m")
    print(f"  tr(D) range            : [{trD.min():.1f}, {trD.max():.1f}] Ns/m")

    ramp_mask = d_human < HUMAN_RAMP_RAD
    near_mask = d_human < HUMAN_COMFORT_RAD
    if np.any(ramp_mask):
        kr = K_diag[ramp_mask]
        print(f"  Ramp-zone K (d < {HUMAN_RAMP_RAD:.2f} m):")
        for i, ax_lbl in enumerate(["xx", "yy", "zz"]):
            print(f"    K_{ax_lbl}: [{kr[:,i].min():.0f}, {kr[:,i].max():.0f}] N/m")
    if np.any(near_mask):
        kn = K_diag[near_mask]
        print(f"  Inside comfort K (d < {HUMAN_COMFORT_RAD:.2f} m):")
        for i, ax_lbl in enumerate(["xx", "yy", "zz"]):
            print(f"    K_{ax_lbl}: [{kn[:,i].min():.0f}, {kn[:,i].max():.0f}] N/m  (target < {K_AXIS_LIMIT:.0f})")
    print("=" * (50*2 + len(" EXP 1b DIAGNOSTICS ")))


# ── shade helper ──────────────────────────────────────────────────────
def _shade_human_zones(ax, t, d_human):
    for mask, col in [
        (d_human < HUMAN_COMFORT_RAD, C_COMFORT),
        (d_human < HUMAN_BODY_RAD,    C_BODY),
    ]:
        if not np.any(mask):
            continue
        starts = np.where(np.diff(mask.astype(int)) ==  1)[0]
        ends   = np.where(np.diff(mask.astype(int)) == -1)[0]
        if mask[0]:  starts = np.concatenate([[0], starts])
        if mask[-1]: ends   = np.concatenate([ends, [len(mask)-1]])
        for s, e in zip(starts, ends):
            ax.axvspan(t[s], t[e], alpha=0.12, color=col, zorder=0)


# ======================================================================
#  PLOT 1 — 3D workspace
# ======================================================================
def plot_3d_workspace(trace, best_cost, base="exp1b_workspace"):
    pos = trace.position

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    cz = HUMAN[2]
    n_seg   = 40
    theta_c = np.linspace(0, 2*np.pi, n_seg)

    # body cylinder (solid red — HARD)
    h_body = 0.40
    z_bot  = cz - 0.10
    z_top  = z_bot + h_body
    xc = HUMAN[0] + HUMAN_BODY_RAD * np.cos(theta_c)
    yc = HUMAN[1] + HUMAN_BODY_RAD * np.sin(theta_c)
    for zb in [z_bot, z_top]:
        ax.plot(xc, yc, [zb]*n_seg, color=C_BODY, lw=1.0, alpha=0.80)
    Th, Zg = np.meshgrid(theta_c, np.linspace(z_bot, z_top, 2))
    ax.plot_surface(HUMAN[0] + HUMAN_BODY_RAD*np.cos(Th),
                    HUMAN[1] + HUMAN_BODY_RAD*np.sin(Th), Zg,
                    alpha=0.35, color=C_BODY, edgecolor="none")

    # comfort zone cylinder (dashed orange — SOFT)
    z_cbot = cz - 0.05
    z_ctop = z_cbot + 0.20
    xcc = HUMAN[0] + HUMAN_COMFORT_RAD * np.cos(theta_c)
    ycc = HUMAN[1] + HUMAN_COMFORT_RAD * np.sin(theta_c)
    for zb in [z_cbot, z_ctop]:
        ax.plot(xcc, ycc, [zb]*n_seg, color=C_COMFORT, lw=0.8, alpha=0.50, ls="--")
    Thc, Zgc = np.meshgrid(theta_c, np.linspace(z_cbot, z_ctop, 2))
    ax.plot_surface(HUMAN[0] + HUMAN_COMFORT_RAD*np.cos(Thc),
                    HUMAN[1] + HUMAN_COMFORT_RAD*np.sin(Thc), Zgc,
                    alpha=0.07, color=C_COMFORT, edgecolor="none")

    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], [START[2], GOAL[2]],
            "--", color=C_DASH, lw=1.6, alpha=0.55, label="Shortest path", zorder=2)
    ax.plot(pos[:,0], pos[:,1], pos[:,2],
            color=C_CARRY, lw=2.5, solid_capstyle="round", zorder=5, label="Trajectory (4 s)")
    ax.scatter(*START, s=70, c=C_START, zorder=10, depthshade=False,
               edgecolors="black", linewidth=0.6)
    ax.scatter(*GOAL,  s=70, c=C_GOAL,  zorder=10, depthshade=False,
               marker="D", edgecolors="black", linewidth=0.6)
    ax.text(HUMAN[0], HUMAN[1]+HUMAN_BODY_RAD+0.01, HUMAN[2]+0.25,
            "Human", fontsize=8, color=C_BODY, ha="center", fontweight="bold")

    ax.set_xlabel("X — forward (m)", fontsize=9, labelpad=7)
    ax.set_ylabel("Y — lateral (m)", fontsize=9, labelpad=7)
    ax.set_zlabel("Z — height (m)",  fontsize=9, labelpad=7)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=22, azim=-50)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("lightgrey")
    ax.grid(True, alpha=0.28)

    extra = [
        mpatches.Patch(facecolor=C_BODY,    alpha=0.40, edgecolor=C_BODY,
                       label=f"Body exclusion — HARD ($r$={HUMAN_BODY_RAD:.2f} m)"),
        mpatches.Patch(facecolor=C_COMFORT, alpha=0.20, edgecolor=C_COMFORT,
                       label=f"Comfort zone — SOFT ($r$={HUMAN_COMFORT_RAD:.2f} m)"),
        mpatches.Patch(facecolor=C_START,   alpha=0.80, edgecolor="black", label="Start"),
        mpatches.Patch(facecolor=C_GOAL,    alpha=0.80, edgecolor="black", label="Goal"),
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
#  PLOT 2 — 2D top-down
# ======================================================================
def plot_2d_topdown(trace, best_cost, base="exp1b_topdown"):
    pos = trace.position

    fig, ax = plt.subplots(figsize=(7, 6.5))

    # comfort zone (soft — dashed orange)
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_COMFORT_RAD,
                             color=C_COMFORT, alpha=0.15, zorder=1))
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_COMFORT_RAD,
                             color=C_COMFORT, fill=False,
                             linestyle="--", linewidth=1.5, zorder=2, alpha=0.85))

    # body exclusion (hard — solid red)
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_BODY_RAD,
                             color=C_BODY, alpha=0.40, zorder=3))
    ax.add_patch(plt.Circle((HUMAN[0], HUMAN[1]), HUMAN_BODY_RAD,
                             color=C_BODY, fill=False,
                             linestyle="-", linewidth=1.8, zorder=4, alpha=1.0))

    ax.scatter(HUMAN[0], HUMAN[1], s=60, c=C_COMFORT, edgecolors=C_BODY,
               linewidth=1.5, zorder=6)
    ax.text(HUMAN[0]+0.01, HUMAN[1]+HUMAN_COMFORT_RAD+0.02,
            "Human", fontsize=8, ha="center", color=C_BODY, fontweight="bold")

    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]],
            "--", color=C_DASH, lw=1.5, alpha=0.50, zorder=3, label="Shortest path")
    ax.plot(pos[:,0], pos[:,1],
            color=C_CARRY, lw=2.5, solid_capstyle="round", zorder=6,
            label="Trajectory (4 s)")

    ax.scatter(START[0], START[1], s=80, c=C_START, zorder=10,
               edgecolors="black", linewidth=0.7, label="Start")
    ax.scatter(GOAL[0],  GOAL[1],  s=75, c=C_GOAL,  zorder=10,
               marker="D", edgecolors="black", linewidth=0.7, label="Goal")

    ax.set_xlabel("$x$ (m)", fontsize=12)
    ax.set_ylabel("$y$ (m)", fontsize=12)
    ax.set_title("Top-down View — Comfort Zone Violation (T=4 s)", fontsize=11)
    ax.set_aspect("equal")
    margin = 0.12
    ax.set_xlim(min(START[0], GOAL[0]) - margin, max(START[0], GOAL[0]) + margin)
    ax.set_ylim(min(START[1], GOAL[1]) - margin, max(START[1], GOAL[1]) + margin)
    ax.grid(True, alpha=0.25)

    legend_extra = [
        mpatches.Patch(facecolor=C_BODY,    alpha=0.45, edgecolor=C_BODY,
                       label=f"Body excl. — HARD ($r$={HUMAN_BODY_RAD:.2f} m)"),
        mpatches.Patch(facecolor=C_COMFORT, alpha=0.20, edgecolor=C_COMFORT,
                       label=f"Comfort zone — SOFT ($r$={HUMAN_COMFORT_RAD:.2f} m)"),
    ]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_extra, fontsize=8.5, loc="lower right",
              framealpha=0.92, edgecolor="lightgrey", fancybox=False)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ======================================================================
#  PLOT 3 — per-axis stiffness vs time
# ======================================================================
def plot_stiffness(trace, best_cost, base="exp1b_stiffness"):
    K_arr   = trace.gains["K"]
    Kd      = np.array([[K[0,0], K[1,1], K[2,2]] for K in K_arr])
    t       = trace.time
    d_human = _distance_to_human(trace.position)

    fig, ax = plt.subplots(figsize=(10, 4.2))

    for mask, col, alpha, lbl in [
        (d_human < HUMAN_RAMP_RAD,    C_COMFORT, 0.07,
         f"Stiffness ramp (d < {HUMAN_RAMP_RAD:.2f} m)"),
        (d_human < HUMAN_COMFORT_RAD, C_COMFORT, 0.14,
         f"Comfort zone (d < {HUMAN_COMFORT_RAD:.2f} m)"),
        (d_human < HUMAN_BODY_RAD,    C_BODY,    0.20,
         f"Body zone (d < {HUMAN_BODY_RAD:.2f} m)"),
    ]:
        if not np.any(mask):
            continue
        starts = np.where(np.diff(mask.astype(int)) ==  1)[0]
        ends   = np.where(np.diff(mask.astype(int)) == -1)[0]
        if mask[0]:  starts = np.concatenate([[0], starts])
        if mask[-1]: ends   = np.concatenate([ends, [len(mask)-1]])
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(t[s], t[e], alpha=alpha, color=col,
                       label=lbl if i == 0 else None, zorder=0)

    ax.plot(t, Kd[:,0], color=C_KX, lw=2.0, label=r"$K_{xx}$")
    ax.plot(t, Kd[:,1], color=C_KY, lw=2.0, label=r"$K_{yy}$")
    ax.plot(t, Kd[:,2], color=C_KZ, lw=2.0, label=r"$K_{zz}$")
    ax.axhline(K_AXIS_LIMIT, color="#CC4444", ls="--", lw=1.4, alpha=0.75,
               label=f"$K_{{\\rm axis}}$ target ≤ {K_AXIS_LIMIT:.0f} N/m")

    ax.set_ylim(0, Kd.max() * 1.15)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Stiffness (N/m)", fontsize=11)
    ax.set_title("Per-axis Stiffness — Compliance Near Human (T=4 s)", fontsize=11)
    ax.set_xlim(0, HORIZON)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ======================================================================
#  PLOT 4 — Euler orientation vs time
# ======================================================================
def plot_orientation_euler(trace, best_cost, base="exp1b_orientation"):
    if trace.orientation is None:
        print("No orientation — skipping.")
        return
    q     = trace.orientation
    t     = trace.time
    euler = np.array([quat_to_euler(q[k]) for k in range(len(t))])
    roll, pitch, yaw = euler[:,0], euler[:,1], euler[:,2]

    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(t, roll,  color=C_KX, lw=1.4, ls="--", alpha=0.60, label=r"Roll $\theta_x$")
    ax.plot(t, yaw,   color=C_KZ, lw=1.4, ls="--", alpha=0.60, label=r"Yaw $\theta_z$")
    ax.plot(t, pitch, color=C_KY, lw=2.2, label=r"Pitch $\theta_y$")
    ax.axhline(0.0, color="#999999", ls="-", lw=0.5, alpha=0.30)
    lim_deg = np.degrees(0.15)
    ax.axhline( lim_deg, color="#CC4444", ls=":", lw=1.0, alpha=0.65,
               label=f"±{lim_deg:.1f}° limit")
    ax.axhline(-lim_deg, color="#CC4444", ls=":", lw=1.0, alpha=0.65)
    all_max = max(abs(roll).max(), abs(pitch).max(), abs(yaw).max(), lim_deg * 1.5)
    ax.set_ylim(-all_max * 1.2, all_max * 1.2)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Angle (deg)", fontsize=11)
    ax.set_title("End-Effector Orientation — Euler Angles (T=4 s)", fontsize=11)
    ax.set_xlim(0, HORIZON)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ======================================================================
#  PLOT 5 — position & velocity kinematics
# ======================================================================
def plot_kinematics(trace, best_cost, base="exp1b_kinematics"):
    pos   = trace.position
    vel   = trace.velocity
    t     = trace.time
    speed = np.linalg.norm(vel, axis=1)
    d_human = _distance_to_human(pos)
    d_goal  = np.linalg.norm(pos - GOAL,  axis=1)

    c_x = "#4C72B0"; c_y = "#DD8452"; c_z = "#55A868"; c_spd = "#8172B2"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axi in axes:
        _shade_human_zones(axi, t, d_human)

    ax0, ax1 = axes
    ax0.plot(t, pos[:,0], color=c_x, lw=1.8, label=r"$x(t)$")
    ax0.plot(t, pos[:,1], color=c_y, lw=1.8, label=r"$y(t)$")
    ax0.plot(t, pos[:,2], color=c_z, lw=1.8, label=r"$z(t)$")
    for val, col, ls in [
        (START[0], c_x, ":"), (START[1], c_y, ":"), (START[2], c_z, ":"),
        (GOAL[0],  c_x, "--"), (GOAL[1],  c_y, "--"), (GOAL[2],  c_z, "--"),
    ]:
        ax0.axhline(val, color=col, lw=0.6, ls=ls, alpha=0.30)

    ax0_r = ax0.twinx()
    ax0_r.plot(t, d_human, color=C_BODY,    lw=1.0, ls=":", alpha=0.60, label="d(human)")
    ax0_r.plot(t, d_goal,  color=C_GOAL,    lw=1.0, ls=":", alpha=0.60, label="d(goal)")
    ax0_r.axhline(HUMAN_BODY_RAD,    color=C_BODY,    lw=0.7, ls="-", alpha=0.40)
    ax0_r.axhline(HUMAN_COMFORT_RAD, color=C_COMFORT, lw=0.7, ls="-", alpha=0.40)
    ax0_r.set_ylabel("Distance (m)", fontsize=9, color="#777777")
    ax0_r.tick_params(labelsize=8)
    ax0_r.legend(fontsize=7.5, loc="center right", framealpha=0.85,
                 edgecolor="lightgrey", fancybox=False)

    ax0.set_ylabel("Position (m)", fontsize=11)
    ax0.legend(fontsize=8.5, loc="upper right", framealpha=0.9,
               edgecolor="lightgrey", fancybox=False, ncol=3)

    ax1.plot(t, vel[:,0], color=c_x,  lw=1.5, alpha=0.75, label=r"$\dot x$")
    ax1.plot(t, vel[:,1], color=c_y,  lw=1.5, alpha=0.75, label=r"$\dot y$")
    ax1.plot(t, vel[:,2], color=c_z,  lw=1.5, alpha=0.75, label=r"$\dot z$")
    ax1.plot(t, speed,    color=c_spd, lw=2.2, label=r"$\|\dot p\|$")
    ax1.axhline( 0.8, color="#333333", ls=":", lw=1.2, alpha=0.65, label="$v_{max}$=0.8")
    ax1.axhline(-0.8, color="#333333", ls=":", lw=1.2, alpha=0.65)
    ax1.axhline( 0.0, color="#999999", ls="-", lw=0.5, alpha=0.30)
    vhi = max(speed.max(), 0.85) * 1.20
    ax1.set_ylim(-vhi, vhi)
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Velocity (m/s)", fontsize=11)
    ax1.set_xlim(0, HORIZON)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8.5, loc="upper right", framealpha=0.9,
               edgecolor="lightgrey", fancybox=False, ncol=4)

    fig.suptitle("Per-axis Position & Velocity (T=4 s)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# =====================================================================
#  PLOT 6 — obstacle HOCBF diagnostics
# =====================================================================
def plot_obstacle_hocbf(trace, best_cost, base="exp1b_obstacle_hocbf"):
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
    ax0, ax1 = axes

    ax0.plot(t, h, color="#4C72B0", lw=2.0, label=r"$h(x)$")
    ax0.axhline(0.0, color="#CC4444", ls=":", lw=1.0, alpha=0.80,
                label=r"$h=0$ (boundary)")
    ax0.set_ylabel("$h(x)$", fontsize=11)
    ax0.set_title("Obstacle HOCBF diagnostics", fontsize=11)
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
    ax1.set_xlim(0, HORIZON)
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
def save_trajectory_csv(trace, csv_path="exp1b_trajectory.csv"):
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
    global START, GOAL, HUMAN
    global HUMAN_BODY_RAD, HUMAN_COMFORT_RAD, HUMAN_RAMP_RAD, HUMAN_GEOMETRY
    global HORIZON

    taskspec = load_taskspec_from_json("spec/exp1b_task.json")
    assert taskspec.phases is not None

    START = np.asarray(taskspec.phases[0]["start"], dtype=float)
    GOAL = np.asarray(taskspec.phases[-1]["end"], dtype=float)
    HORIZON = float(np.sum([float(p["duration"]) for p in taskspec.phases]))

    body_clause = next((cl for cl in taskspec.clauses if cl.predicate == "HumanBodyExclusion"), None)
    if body_clause is None:
        raise ValueError("exp1b_task.json must include a HumanBodyExclusion clause")
    if "human_position" not in body_clause.parameters or "body_radius" not in body_clause.parameters:
        raise ValueError("HumanBodyExclusion must define human_position and body_radius")

    HUMAN = np.asarray(body_clause.parameters["human_position"], dtype=float)
    HUMAN_BODY_RAD = float(body_clause.parameters["body_radius"])
    HUMAN_GEOMETRY = str(body_clause.parameters.get("geometry", "sphere"))

    comfort_clause = next((cl for cl in taskspec.clauses if cl.predicate == "HumanComfortDistance"), None)
    if comfort_clause is not None:
        if "preferred_distance" in comfort_clause.parameters:
            HUMAN_COMFORT_RAD = float(comfort_clause.parameters["preferred_distance"])
        if "human_position" in comfort_clause.parameters:
            HUMAN = np.asarray(comfort_clause.parameters["human_position"], dtype=float)
    HUMAN_RAMP_RAD = 3.0 * HUMAN_COMFORT_RAD

    policy = MultiPhaseCertifiedPolicy(taskspec.phases, K0=300.0, D0=30.0)

    # Layers 1+2 (DMP repulsion + radial projector) are wired automatically
    # from the JSON modality="HARD" clause — no manual hardcoding needed.
    policy.setup_hard_obstacles_from_taskspec(taskspec)

    theta_dim = policy.parameter_dimension()
    print(f"Scene 5b: Carry to Goal in {HORIZON} s (human at {HUMAN.tolist()})")
    print(f"  theta_dim={theta_dim}")
    print(f"  Body excl. (HARD, from JSON):  geometry={HUMAN_GEOMETRY}, r={HUMAN_BODY_RAD:.2f} m")
    if comfort_clause is not None:
        print(f"  Comfort zone (SOFT):  r={HUMAN_COMFORT_RAD:.2f} m  — PREFER weight={comfort_clause.weight:.1f}")
    else:
        print(f"  Comfort zone (SOFT):  r={HUMAN_COMFORT_RAD:.2f} m")
    print(f"  Stiffness ramp:       starts at d={HUMAN_RAMP_RAD:.2f} m")

    predicate_registry = build_predicate_registry()
    compiler = Compiler(predicate_registry,
                        human_position=HUMAN,
                        human_proximity_radius=HUMAN_COMFORT_RAD)
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

    optimizer = PIBB(theta=theta_init, sigma=sigma_init, beta=8.0, decay=0.99)

    N_SAMPLES = 30
    N_UPDATES = 70
    best_cost  = float("inf")
    best_theta = theta_init.copy()

    print(f"\nPIBB: {N_UPDATES} updates × {N_SAMPLES} samples")

    for upd in range(N_UPDATES):
        samples = optimizer.sample(N_SAMPLES)
        costs   = np.array([objective_fn(policy.rollout(samples[i]))
                            for i in range(N_SAMPLES)])
        costs_s = np.clip(np.where(np.isfinite(costs), costs, 1e4), 0.0, 1e4)
        optimizer.update(samples, costs_s)

        if costs.min() < best_cost:
            best_cost  = costs.min()
            best_theta = samples[np.argmin(costs)].copy()

        print(f"  [{upd+1:03d}]  min={costs.min():.4f}  "
              f"mean={costs.mean():.4f}  best={best_cost:.4f}")

    print("Optimization complete.\n")

    trace_final = policy.rollout(best_theta)
    save_trajectory_csv(trace_final, csv_path="exp1b_trajectory.csv")
    print_diagnostics(trace_final, best_cost)

    plot_3d_workspace(trace_final, best_cost)
    plot_2d_topdown(trace_final, best_cost)
    plot_stiffness(trace_final, best_cost)
    plot_orientation_euler(trace_final, best_cost)
    plot_kinematics(trace_final, best_cost)
    plot_obstacle_hocbf(trace_final, best_cost)

    print("Exp 1b done — 6 plots saved as PNG.")


if __name__ == "__main__":
    main()
