"""
main_llm.py — Language-Conditioned Trajectory Synthesis via LLM

USAGE
-----
  conda activate clavic
  export GEMINI_API_KEY="AIza..."   # get free at https://aistudio.google.com/app/apikey
  python main_llm.py

  # Or pass a task description as a command-line argument:
  python main_llm.py "Carry the mug from [0.55,0,0.3] to [0.3,0.55,0.3] in 8s.
                      A human is standing at [0.3,0.3,0.3]. Must not touch the human.
                      Keep velocity below 0.5 m/s."

HOW THIS WORKS
--------------
Step 1  — LLMAgent.generate(task_description)
            Sends a carefully constructed system prompt (catalogue + weight
            rules + modality rules + few-shot examples) to GPT-4o at
            temperature=0.2. Parses the JSON response.

Step 2  — validate_and_clamp()   [inside LLMAgent — automatic]
            Two-pass firewall:
              Pass 1 HARD REJECT : unknown predicate, illegal modality,
                                   missing required binding → retry w/ error feedback
              Pass 2 SILENT CLAMP: weight out of [1,20], param out of catalogue
                                   [min,max] → auto-fixed, warning printed.
            Up to 3 retries with error message fed back to the LLM.

Step 3  — json_parser.load_taskspec_from_dict()
            Converts validated dict → TaskSpec(clauses, phases, hard_obstacle_specs).
            HARD clauses auto-populate hard_obstacle_specs (used by Layers 1+2).

Step 4  — MultiPhaseCertifiedPolicy + setup_hard_obstacles_from_taskspec()
            Wires DMP repulsion (Layer 1) + radial projector (Layer 2) from
            hard_obstacle_specs. Nothing hard-coded — comes entirely from JSON.

Step 5  — Compiler.compile(taskspec) → objective_fn(trace)
            HARD/REQUIRE  → SLACK_WEIGHT=500 * max(0, -rho)²
            PREFER        → clause.weight   * max(0, -rho)
            + intrinsic costs: K-ceiling, D-min, human-proximity stiffness shaping

Step 6  — PIBB optimiser (70 updates × 30 samples)
            Saturates at ~60-70 epochs. Outputs best_theta + best_trace.

Step 7  — Diagnostics + plots (seaborn / matplotlib, 300 dpi PNG).

ROBUSTNESS GUARANTEES (what the LLM cannot break)
--------------------------------------------------
  • Predicate whitelist     — any name not in CATALOGUE is hard-rejected
  • Modality locking        — HumanBodyExclusion MUST be HARD; fixed in catalogue
                              and re-enforced in validator (two independent checks)
  • Weight clamping         — weight>20 silently fixed to 20.0 before reaching optimiser
  • Param bounds clamping   — every float binding (radius, vmax …) clipped to [min,max]
  • Operator whitelist      — only {always, eventually, always_during,
                              eventually_during, until} accepted
  • time_window enforcement — *_during operators must supply time_window, else rejected
  • horizon_sec guard       — clamped to [1.0, 60.0]
  • 3-layer geometry        — Layers 1+2 (DMP repulsion + projector) are
                              ALWAYS active for HARD clauses regardless of optimiser
"""

import json
import logging
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from llm_interface.llm_agent import LLMAgent
from spec.json_parser import load_taskspec_from_json
from spec.compiler import Compiler
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
    zero_velocity,
    hold_at_waypoint,
    obstacle_avoidance,
    orientation_at_target,
    orientation_hold,
    directional_stiffness_near_human,
)

# ── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main_llm")

# ── seaborn style ─────────────────────────────────────────────────────────
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# ── colour palette ────────────────────────────────────────────────────────
C_TRAJ    = "#4C72B0"
C_BODY    = "#E74C3C"
C_COMFORT = "#F39C12"
C_OBS     = "#8B4513"
C_START   = "#2CA02C"
C_GOAL    = "#1F77B4"
C_DASH    = "#999999"
C_KX      = "#4C72B0"
C_KY      = "#DD8452"
C_KZ      = "#55A868"

# ── full predicate registry (every predicate in the catalogue) ────────────
PREDICATE_REGISTRY = {
    "AtGoal":                        at_waypoint,
    "AtWaypoint":                    at_waypoint,
    "HoldAtWaypoint":                hold_at_waypoint,
    "HumanBodyExclusion":            human_body_exclusion,
    "HumanComfortDistance":          human_comfort_distance,
    "ObstacleAvoidance":             obstacle_avoidance,
    "VelocityLimit":                 velocity_limit,
    "AngularVelocityLimit":          angular_velocity_limit,
    "ZeroVelocity":                  zero_velocity,
    "OrientationLimit":              orientation_limit,
    "OrientationAtTarget":           orientation_at_target,
    "OrientationHold":               orientation_hold,
    "DirectionalStiffnessNearHuman": directional_stiffness_near_human,
}


# ─────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────

def _extract_human_info(taskspec):
    """
    Scan hard_obstacle_specs + clause bindings for human position and
    comfort radius — used to pass to Compiler for stiffness shaping.
    Returns (human_pos, comfort_radius) or (None, None).
    """
    human_pos      = None
    comfort_radius = None

    # Human position comes from HumanBodyExclusion binding (HARD clause)
    for clause in taskspec.clauses:
        if clause.predicate == "HumanBodyExclusion":
            hp = clause.parameters.get("human_position")
            if hp is not None:
                human_pos = np.asarray(hp, float)
            break

    # Comfort radius comes from HumanComfortDistance binding (PREFER clause)
    for clause in taskspec.clauses:
        if clause.predicate == "HumanComfortDistance":
            cr = clause.parameters.get("preferred_distance")
            if cr is not None:
                comfort_radius = float(cr)
            break

    return human_pos, comfort_radius


def _spec_dict_to_taskspec(spec_dict: dict):
    """
    Write spec_dict to a temp JSON file, parse with load_taskspec_from_json,
    then clean up. This reuses the full json_parser logic including
    hard_obstacle_specs extraction.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="clavic_llm_"
    ) as f:
        json.dump(spec_dict, f, indent=2)
        tmp_path = f.name
    logger.info("Temporary task JSON written to: %s", tmp_path)

    try:
        taskspec = load_taskspec_from_json(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return taskspec


def _print_spec_summary(spec_dict: dict):
    """Pretty-print the validated spec dict so the user can review it."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  LLM-GENERATED TASK SPEC (validated + clamped)")
    print(sep)
    print(f"  horizon_sec : {spec_dict.get('horizon_sec')} s")
    phases = spec_dict.get("phases", [])
    print(f"  phases      : {len(phases)}")
    for i, ph in enumerate(phases):
        print(f"    [{i}] {ph.get('label','?')}  "
              f"{ph.get('start')} → {ph.get('end')}  "
              f"dur={ph.get('duration')}s")
    print(f"  clauses     : {len(spec_dict.get('clauses', []))}")
    for cl in spec_dict.get("clauses", []):
        print(f"    [{cl['modality']:7s}] {cl['type']:22s} "
              f"{cl['predicate']:30s}  w={cl['weight']}")
    print(f"  bindings    :")
    for k, v in spec_dict.get("bindings", {}).items():
        print(f"    {k}: {v}")
    print(sep + "\n")


def quat_to_euler_deg(q):
    """Convert quaternion [w,x,y,z] to Euler ZYX in degrees."""
    w, x, y, z = q
    roll  = np.degrees(np.arctan2(2*(w*x + y*z),  1 - 2*(x*x + y*y)))
    sinp  = np.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y),  1 - 2*(y*y + z*z)))
    return roll, pitch, yaw


def save_trajectory_csv(trace, csv_path="llm_trajectory.csv"):
    """Save trajectory to 25-column CSV: pos + quat + full K + full D."""
    import csv

    pos = trace.position
    T = len(trace.time)
    if trace.orientation is not None and len(trace.orientation) == T:
        quat = trace.orientation
    else:
        quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))

    K_arr = trace.gains["K"]
    D_arr = trace.gains["D"]

    header = [
        "x", "y", "z", "qw", "qx", "qy", "qz",
        "k11", "k12", "k13", "k21", "k22", "k23", "k31", "k32", "k33",
        "d11", "d12", "d13", "d21", "d22", "d23", "d31", "d32", "d33",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(T):
            K = K_arr[i]
            D = D_arr[i]
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


# ─────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────

def _draw_sphere_circles(ax, center, radius, color, alpha=0.35, label=None):
    """Draw 3 orthogonal circles on a 3D axis to hint at a sphere."""
    n = 60
    th = np.linspace(0, 2*np.pi, n)
    cx, cy, cz = center
    r = radius
    kw = dict(color=color, lw=1.2, alpha=alpha)
    for i, (a, b, fixed, val) in enumerate([
        (cx + r*np.cos(th), cy + r*np.sin(th), "z", cz),
        (cx + r*np.cos(th), np.full(n, cy),    "y", cy),  # xz
        (np.full(n, cx),    cy + r*np.cos(th), "x", cx),  # yz
    ]):
        if i == 0:
            ax.plot(cx + r*np.cos(th), cy + r*np.sin(th), cz, label=label, **kw)
        elif i == 1:
            ax.plot(cx + r*np.cos(th), np.full(n, cy), cz + r*np.sin(th), **kw)
        else:
            ax.plot(np.full(n, cx), cy + r*np.cos(th), cz + r*np.sin(th), **kw)


def plot_workspace_3d(trace, spec_dict, taskspec, base="llm_workspace"):
    pos    = trace.position
    phases = spec_dict.get("phases", [])
    title  = "LLM-Generated Trajectory — 3D Workspace"

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # ── draw hard obstacle spheres ──
    for obs in taskspec.hard_obstacle_specs:
        c = obs["center"]
        r = obs["radius"]
        _draw_sphere_circles(ax, c, r, C_BODY, alpha=0.55,
                             label=f"HARD obstacle (r={r:.2f} m)")

    # ── draw PREFER human comfort zone if any ──
    for clause in taskspec.clauses:
        if clause.predicate == "HumanComfortDistance":
            hp = clause.parameters.get("human_position")
            cr = clause.parameters.get("preferred_distance")
            if hp is not None and cr is not None:
                hp = np.asarray(hp, float)
                _draw_sphere_circles(ax, hp, cr, C_COMFORT, alpha=0.25,
                                     label=f"Comfort zone (r={cr:.2f} m)")
            break

    # ── straight-line start → end ──
    if phases:
        s = np.array(phases[0]["start"])
        e = np.array(phases[-1]["end"])
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
                "--", color=C_DASH, lw=1.5, alpha=0.50, label="Direct path", zorder=2)

    # ── trajectory ──
    ax.plot(pos[:,0], pos[:,1], pos[:,2],
            color=C_TRAJ, lw=2.5, solid_capstyle="round", zorder=5, label="Trajectory")

    # ── start / goal markers ──
    for ph in phases:
        s = np.array(ph["start"])
        e = np.array(ph["end"])
        ax.scatter(*s, s=70, c=C_START, zorder=10, depthshade=False,
                   edgecolors="black", linewidth=0.6)
        ax.scatter(*e, s=70, c=C_GOAL, zorder=10, depthshade=False,
                   marker="D", edgecolors="black", linewidth=0.6)

    ax.set_xlabel("X (m)", fontsize=9, labelpad=7)
    ax.set_ylabel("Y (m)", fontsize=9, labelpad=7)
    ax.set_zlabel("Z (m)", fontsize=9, labelpad=7)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=22, azim=-50)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("lightgrey")
    ax.grid(True, alpha=0.28)

    handles, _ = ax.get_legend_handles_labels()
    handles += [
        mpatches.Patch(facecolor=C_START, alpha=0.8, edgecolor="black", label="Phase start"),
        mpatches.Patch(facecolor=C_GOAL,  alpha=0.8, edgecolor="black", label="Phase end"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.0, 0.97), framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)
    ax.set_title(title, fontsize=9, pad=10)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


def plot_topdown(trace, spec_dict, taskspec, base="llm_topdown"):
    pos    = trace.position
    phases = spec_dict.get("phases", [])

    fig, ax = plt.subplots(figsize=(7, 6.5))

    # ── hard obstacle circles ──
    for obs in taskspec.hard_obstacle_specs:
        cx, cy = obs["center"][0], obs["center"][1]
        r = obs["radius"]
        ax.add_patch(plt.Circle((cx, cy), r, color=C_BODY, alpha=0.40, zorder=3))
        ax.add_patch(plt.Circle((cx, cy), r, color=C_BODY, fill=False,
                                linewidth=2.0, zorder=4))
        ax.text(cx, cy, "Obstacle", fontsize=7, ha="center", va="center",
                color="white", fontweight="bold", zorder=5)

    # ── comfort zone ──
    for clause in taskspec.clauses:
        if clause.predicate == "HumanComfortDistance":
            hp = clause.parameters.get("human_position")
            cr = clause.parameters.get("preferred_distance")
            if hp is not None and cr is not None:
                hx, hy = float(hp[0]), float(hp[1])
                ax.add_patch(plt.Circle((hx, hy), cr, color=C_COMFORT,
                                        alpha=0.14, zorder=1))
                ax.add_patch(plt.Circle((hx, hy), cr, color=C_COMFORT,
                                        fill=False, linestyle="--",
                                        linewidth=1.8, zorder=2, alpha=0.90))
            break

    # ── straight start → goal ──
    if phases:
        s = phases[0]["start"];  e = phases[-1]["end"]
        ax.plot([s[0], e[0]], [s[1], e[1]],
                "--", color=C_DASH, lw=1.5, alpha=0.50, zorder=3, label="Direct path")

    # ── trajectory ──
    ax.plot(pos[:,0], pos[:,1], color=C_TRAJ, lw=2.5,
            solid_capstyle="round", zorder=6, label="Trajectory")

    # ── start / goal markers ──
    for ph in phases:
        s = ph["start"]; e = ph["end"]
        ax.scatter(s[0], s[1], s=80, c=C_START, zorder=10,
                   edgecolors="black", linewidth=0.7)
        ax.scatter(e[0], e[1], s=75, c=C_GOAL, zorder=10,
                   marker="D", edgecolors="black", linewidth=0.7)

    ax.set_xlabel("$x$ (m)", fontsize=12)
    ax.set_ylabel("$y$ (m)", fontsize=12)
    ax.set_title("Top-down View — LLM Task", fontsize=11)
    ax.set_aspect("equal")

    all_pts = np.array([ph["start"] for ph in phases] + [ph["end"] for ph in phases])
    margin = 0.15
    ax.set_xlim(all_pts[:,0].min() - margin, all_pts[:,0].max() + margin)
    ax.set_ylim(all_pts[:,1].min() - margin, all_pts[:,1].max() + margin)
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        mpatches.Patch(facecolor=C_BODY,    alpha=0.45, edgecolor=C_BODY,
                       label="HARD obstacle"),
        mpatches.Patch(facecolor=C_COMFORT, alpha=0.20, edgecolor=C_COMFORT,
                       label="Comfort zone (SOFT)"),
        mpatches.Patch(facecolor=C_START,   alpha=0.8,  edgecolor="black",
                       label="Phase start"),
        mpatches.Patch(facecolor=C_GOAL,    alpha=0.8,  edgecolor="black",
                       label="Phase end"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right",
              framealpha=0.92, edgecolor="lightgrey", fancybox=False)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


def plot_stiffness(trace, spec_dict, taskspec, horizon, base="llm_stiffness"):
    if trace.gains is None or "K" not in trace.gains:
        print("No stiffness data — skipping stiffness plot.")
        return
    K_arr = trace.gains["K"]
    Kd    = np.array([[K[0,0], K[1,1], K[2,2]] for K in K_arr])
    t     = trace.time

    # Find human position from spec if any (for shading)
    human_pos = None
    human_cr  = None
    for clause in taskspec.clauses:
        if clause.predicate == "HumanBodyExclusion":
            hp = clause.parameters.get("human_position")
            if hp is not None:
                human_pos = np.asarray(hp, float)
        if clause.predicate == "HumanComfortDistance":
            cr = clause.parameters.get("preferred_distance")
            if cr is not None:
                human_cr = float(cr)

    fig, ax = plt.subplots(figsize=(10, 4.2))

    # ── shade near-human region ──
    if human_pos is not None and human_cr is not None:
        d_h = np.linalg.norm(trace.position - human_pos, axis=1)
        for mask, col, lbl in [
            (d_h < 3*human_cr, C_COMFORT, f"Stiffness ramp zone (d < {3*human_cr:.2f} m)"),
            (d_h < human_cr,   C_COMFORT, f"Comfort zone (d < {human_cr:.2f} m)"),
        ]:
            if np.any(mask):
                starts = np.where(np.diff(mask.astype(int)) ==  1)[0]
                ends   = np.where(np.diff(mask.astype(int)) == -1)[0]
                if mask[0]:  starts = np.concatenate([[0], starts])
                if mask[-1]: ends   = np.concatenate([ends, [len(mask)-1]])
                for i, (s, e) in enumerate(zip(starts, ends)):
                    ax.axvspan(t[s], t[e], alpha=0.10, color=col,
                               label=lbl if i == 0 else None, zorder=0)

    ax.plot(t, Kd[:,0], color=C_KX, lw=2.0, label=r"$K_{xx}$")
    ax.plot(t, Kd[:,1], color=C_KY, lw=2.0, label=r"$K_{yy}$")
    ax.plot(t, Kd[:,2], color=C_KZ, lw=2.0, label=r"$K_{zz}$")

    ax.set_ylim(0, max(Kd.max() * 1.15, 50))
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Stiffness (N/m)", fontsize=11)
    ax.set_title("Per-axis Stiffness Schedule — LLM Task", fontsize=11)
    ax.set_xlim(0, horizon)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


def plot_orientation(trace, horizon, base="llm_orientation"):
    if trace.orientation is None:
        print("No orientation — skipping orientation plot.")
        return
    q     = trace.orientation
    t     = trace.time
    euler = np.array([quat_to_euler_deg(q[k]) for k in range(len(t))])
    roll, pitch, yaw = euler[:,0], euler[:,1], euler[:,2]

    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(t, roll,  color=C_KX, lw=1.4, ls="--", alpha=0.60, label=r"Roll $\theta_x$")
    ax.plot(t, yaw,   color=C_KZ, lw=1.4, ls="--", alpha=0.60, label=r"Yaw $\theta_z$")
    ax.plot(t, pitch, color=C_KY, lw=2.2, label=r"Pitch $\theta_y$")
    ax.axhline(0.0, color="#999999", ls="-", lw=0.5, alpha=0.30)

    all_max = max(abs(roll).max(), abs(pitch).max(), abs(yaw).max(), 5.0)
    ax.set_ylim(-all_max * 1.3, all_max * 1.3)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Angle (deg)", fontsize=11)
    ax.set_title("End-Effector Orientation — Euler Angles", fontsize=11)
    ax.set_xlim(0, horizon)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)
    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


def plot_kinematics(trace, horizon, base="llm_kinematics"):
    pos   = trace.position
    vel   = trace.velocity
    t     = trace.time
    speed = np.linalg.norm(vel, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # position
    ax = axes[0]
    ax.plot(t, pos[:,0], color=C_KX, lw=1.8, label="$p_x$")
    ax.plot(t, pos[:,1], color=C_KY, lw=1.8, label="$p_y$")
    ax.plot(t, pos[:,2], color=C_KZ, lw=1.8, label="$p_z$")
    ax.set_ylabel("Position (m)", fontsize=11)
    ax.set_title("Kinematics — LLM Task", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              edgecolor="lightgrey", fancybox=False)
    ax.grid(True, alpha=0.25)

    # speed
    ax2 = axes[1]
    ax2.plot(t, speed, color="#8172B2", lw=2.0, label="Speed $\\|\\dot{p}\\|$")
    ax2.set_ylabel("Speed (m/s)", fontsize=11)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_xlim(0, horizon)
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.9,
               edgecolor="lightgrey", fancybox=False)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {base}.png")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────
#  Diagnostics
# ─────────────────────────────────────────────────────────────────────────

def print_diagnostics(trace, taskspec, best_cost):
    pos   = trace.position
    speed = np.linalg.norm(trace.velocity, axis=1)
    K_arr = trace.gains["K"] if trace.gains else None

    sep = "=" * 58
    print(f"\n{sep}")
    print("  LLM TASK DIAGNOSTICS")
    print(sep)
    print(f"  Best cost        : {best_cost:.4f}")
    print(f"  Max speed        : {speed.max():.4f} m/s")

    if K_arr is not None:
        trK = np.array([np.trace(K) for K in K_arr])
        print(f"  tr(K) range      : [{trK.min():.0f}, {trK.max():.0f}] N/m")

    # Obstacle clearances
    for obs in taskspec.hard_obstacle_specs:
        c = np.asarray(obs["center"], float)
        r = obs["radius"]
        d_obs = np.linalg.norm(pos - c, axis=1)
        clearance_cm = (d_obs.min() - r) * 100.0
        n_viol = int(np.sum(d_obs < r))
        label = "human body" if obs.get("avoidance") == "HARD" and r <= 0.10 else "obstacle"
        print(f"  HARD [{label:12s}] r={r:.2f}m  "
              f"min_clearance={clearance_cm:+.1f}cm  "
              f"violations={n_viol}  (must be 0)")

    # Goal reached?
    for clause in taskspec.clauses:
        if clause.predicate in ("AtGoal", "AtWaypoint"):
            wp  = np.asarray(clause.parameters.get("waypoint", [0,0,0]), float)
            tol = float(clause.parameters.get("tolerance", 0.03))
            d_g = np.linalg.norm(pos - wp, axis=1)
            reached = bool(np.any(d_g < tol))
            print(f"  Goal reached     : {'YES' if reached else 'NO'}  "
                  f"(tolerance={tol:.3f} m)")
            break

    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    # ══════════════════════════════════════════════════════════════════════
    #  ★ CONFIGURATION — edit this block before running ★
    # ══════════════════════════════════════════════════════════════════════

    # ── API key (paste directly here OR leave "" to use env var) ──────────
    # You can also set it in your shell: export GEMINI_API_KEY="AIza..."
    # Get a free key at: https://aistudio.google.com/app/apikey
    GEMINI_API_KEY_INLINE = "your Gemini API key here (or leave empty to use env var)"  

    # ── Gemini model ──────────────────────────────────────────────────────
    # "gemini-2.0-flash"              — fast, free tier, good for most tasks
    # "gemini-1.5-pro"                — slower, smarter, better for complex multi-phase
    # "gemini-2.5-pro-preview-03-25"  — most capable (may need paid tier)
    GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

    # ── Task to run ───────────────────────────────────────────────────────
    # Set TASK_CHOICE to one of:
    #   "exp1"   — carry + human avoidance  (same as main_exp1.py)
    #   "exp2"   — carry + obstacle avoidance (same as main_exp2.py)
    #   "exp3a"  — multi-phase carry+pour    (same as main_exp3a.py)
    #   "custom" — use CUSTOM_TASK_DESCRIPTION below
    #   "argv"   — read task from command-line argument
    #              python main_llm.py "Your task description here"
    TASK_CHOICE = "argv"

    CUSTOM_TASK_DESCRIPTION = (
        "Carry the mug from position [0.55, 0.0, 0.3] to position [0.3, 0.55, 0.3] "
        "over 10 seconds. A human is standing at position [0.3, 0.3, 0.3]. "
        "The robot must never touch the human body (strict safety). "
        "Prefer to maintain comfortable distance from the human. "
        "Keep end-effector velocity below 0.8 m/s at all times. "
        "Maintain upright orientation (deviation from [1,0,0,0] below 0.3 rad)."
    )

    # ══════════════════════════════════════════════════════════════════════
    #  Predefined task descriptions (mirrors the spec/exp*_task.json files)
    # ══════════════════════════════════════════════════════════════════════
    _PREDEFINED_TASKS = {
        "exp1": (
            "Carry the mug from position [0.55, 0.0, 0.3] to position [0.3, 0.55, 0.3] "
            "over 10 seconds. "
            "A human is standing at position [0.3, 0.3, 0.3]. "
            "The robot must NEVER touch the human body (hard safety, body radius 0.08 m). "
            "Prefer to maintain a comfortable distance of 0.19 m from the human. "
            "Reduce stiffness when close to the human (DirectionalStiffnessNearHuman). "
            "Keep end-effector velocity below 0.8 m/s at all times. "
            "Maintain upright orientation — deviation from [1,0,0,0] below 0.3 rad."
        ),
        "exp2": (
            "Carry the mug from position [0.55, 0.0, 0.3] to position [0.05, 0.55, 0.3] "
            "over 10 seconds. "
            "There is a cylindrical obstacle at position [0.30, 0.25, 0.3] with radius 0.10 m. "
            "The robot must NEVER collide with the obstacle (hard safety). "
            "Keep end-effector velocity below 0.8 m/s at all times. "
            "Maintain upright orientation — deviation from [1,0,0,0] below 0.3 rad."
        ),
        "exp3a": (
            "Two-phase task over 14 seconds total. "
            "Phase 1 'carry' (8 s): move from [0.55, 0.0, 0.3] to [0.3, 0.55, 0.3]. "
            "Phase 2 'pour' (6 s): move from [0.3, 0.55, 0.3] to [0.3, 0.55, 0.3] (hold), "
            "rotating from upright quaternion [1,0,0,0] to pour quaternion [0.707,0,0.707,0]. "
            "A human is at [0.3, 0.3, 0.3]. Must never touch the human (hard safety, radius 0.08 m). "
            "Prefer comfortable distance 0.19 m from human. "
            "Keep velocity below 0.8 m/s. "
            "During the pour phase (t=8 to t=14), hold at the waypoint [0.3, 0.55, 0.3] "
            "and achieve the pour orientation."
        ),
    }

    # ── 1. Resolve task description ───────────────────────────────────────
    if TASK_CHOICE == "argv":
        if len(sys.argv) < 2:
            print("ERROR: TASK_CHOICE='argv' but no argument given.")
            print("  Usage: python main_llm.py \"Your task description here\"")
            sys.exit(1)
        task_description = " ".join(sys.argv[1:])
    elif TASK_CHOICE == "custom":
        task_description = CUSTOM_TASK_DESCRIPTION
    elif TASK_CHOICE in _PREDEFINED_TASKS:
        task_description = _PREDEFINED_TASKS[TASK_CHOICE]
    else:
        print(f"ERROR: Unknown TASK_CHOICE='{TASK_CHOICE}'. "
              f"Choose from: {list(_PREDEFINED_TASKS.keys()) + ['custom', 'argv']}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  CLAVIC — Language-Conditioned Trajectory Synthesis")
    print("=" * 60)
    print(f"\nTask: {TASK_CHOICE}")
    print(f"Description:\n  {task_description}\n")

    # ── 2. LLM call ───────────────────────────────────────────────────────
    # Priority: inline key → env var
    api_key = GEMINI_API_KEY_INLINE or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: No GEMINI_API_KEY found.")
        print("  Either paste it into GEMINI_API_KEY_INLINE above,")
        print("  or run: export GEMINI_API_KEY='AIza...'")
        print("  Get a free key at: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    print(f"Calling LLM ({GEMINI_MODEL}) ...")
    agent = LLMAgent(model=GEMINI_MODEL, api_key=api_key)

    try:
        spec_dict = agent.generate(task_description)
    except RuntimeError as e:
        print(f"\nLLM failed to produce a valid spec:\n  {e}")
        sys.exit(1)

    # ── 3. Print spec for user review ────────────────────────────────────
    _print_spec_summary(spec_dict)

    # Save the validated spec for reproducibility
    spec_path = "llm_task_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec_dict, f, indent=2)
    print(f"Validated spec saved to: {spec_path}\n")

    # ── 4. Parse into TaskSpec ────────────────────────────────────────────
    taskspec = _spec_dict_to_taskspec(spec_dict)

    horizon = taskspec.horizon_sec
    phases  = taskspec.phases   # list of dicts from JSON

    print(f"TaskSpec parsed: {len(taskspec.clauses)} clauses, "
          f"{len(taskspec.hard_obstacle_specs)} HARD obstacles, "
          f"{len(phases)} phase(s).")

    # ── 5. Build multi-phase policy ───────────────────────────────────────
    policy = MultiPhaseCertifiedPolicy(phases, K0=200.0, D0=30.0)

    # Wire Layers 1+2 automatically from HARD clauses in JSON
    policy.setup_hard_obstacles_from_taskspec(taskspec)

    theta_dim = policy.parameter_dimension()
    print(f"Policy: {len(phases)} phase(s), theta_dim={theta_dim}")

    # ── 6. Build compiler ─────────────────────────────────────────────────
    # Auto-detect human position + comfort radius from spec for stiffness shaping
    human_pos, human_cr = _extract_human_info(taskspec)
    if human_pos is not None:
        print(f"Human detected in spec at {human_pos.tolist()}, "
              f"comfort_radius={human_cr}")

    compiler = Compiler(
        predicate_registry=PREDICATE_REGISTRY,
        human_position=human_pos,
        human_proximity_radius=human_cr,
        k_max_global=3000.0,
    )
    objective_fn = compiler.compile(taskspec)

    # ── 7. Nominal cost check ─────────────────────────────────────────────
    trace0 = policy.rollout(np.zeros(theta_dim))
    cost0  = objective_fn(trace0)
    print(f"Nominal cost (theta=0): {cost0:.4f}")

    # ── 8. PIBB optimisation ──────────────────────────────────────────────
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

    print(f"\nPIBB: {N_UPDATES} updates × {N_SAMPLES} samples/update")
    print("-" * 50)

    for upd in range(N_UPDATES):
        samples = optimizer.sample(N_SAMPLES)
        costs   = np.array([
            objective_fn(policy.rollout(samples[i]))
            for i in range(N_SAMPLES)
        ])
        costs_s = np.clip(np.where(np.isfinite(costs), costs, 1e4), 0.0, 1e4)
        optimizer.update(samples, costs_s)

        if costs.min() < best_cost:
            best_cost  = costs.min()
            best_theta = samples[np.argmin(costs)].copy()

        print(f"  [{upd+1:03d}/{N_UPDATES}]  "
              f"min={costs.min():.4f}  "
              f"mean={costs.mean():.4f}  "
              f"best={best_cost:.4f}")

    print("-" * 50)
    print("Optimisation complete.\n")

    # ── 9. Final rollout & save ───────────────────────────────────────────
    trace_final = policy.rollout(best_theta)
    save_trajectory_csv(trace_final, csv_path="llm_trajectory.csv")
    np.savez("llm_checkpoint.npz", best_theta=best_theta, best_cost=best_cost)
    print(f"Checkpoint saved: llm_checkpoint.npz")

    # ── 10. Diagnostics ───────────────────────────────────────────────────
    print_diagnostics(trace_final, taskspec, best_cost)

    # ── 11. Plots ─────────────────────────────────────────────────────────
    plot_workspace_3d(trace_final, spec_dict, taskspec, base="llm_workspace")
    plot_topdown(trace_final, spec_dict, taskspec,      base="llm_topdown")
    plot_stiffness(trace_final, spec_dict, taskspec, horizon, base="llm_stiffness")
    plot_orientation(trace_final, horizon,               base="llm_orientation")
    plot_kinematics(trace_final, horizon,                base="llm_kinematics")

    print("\nAll done — outputs:")
    print("  llm_task_spec.json   — validated task spec (for reproducibility)")
    print("  llm_trajectory.csv   — full trajectory + gains")
    print("  llm_checkpoint.npz   — best_theta + best_cost")
    print("  llm_workspace.png    — 3D trajectory")
    print("  llm_topdown.png      — top-down view")
    print("  llm_stiffness.png    — per-axis K schedule")
    print("  llm_orientation.png  — Euler angles")
    print("  llm_kinematics.png   — position + speed")


if __name__ == "__main__":
    main()
