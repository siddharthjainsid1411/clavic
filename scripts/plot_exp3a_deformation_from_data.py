#!/usr/bin/env python3
"""
Standalone plotting utility for Exp 3a deformation diagnostics.

This script does not modify or depend on the experiment entrypoint. It can:
  1) plot directly from a saved `exp3a_trajectory.csv`
  2) optionally use a compatible checkpoint to reconstruct the nominal
     pre-deformation path and the projector correction vectors

Examples
--------
python scripts/plot_exp3a_deformation_from_data.py

python scripts/plot_exp3a_deformation_from_data.py \
  --csv exp3a_trajectory.csv \
  --checkpoint some_exp3a_checkpoint.npz \
  --prefix exp3a_debug
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as mpatches
import seaborn as sns

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spec.json_parser import load_taskspec_from_json
from core.multi_phase_policy import MultiPhaseCertifiedPolicy
from core.obstacle_projection import ObstacleProjector


sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

C_ACTUAL = "#4C72B0"
C_ACTIVE = "#D43D3D"
C_NOM = "#8A8A8A"
C_OBS = "#AAAAAA"
C_GOAL = "#1F77B4"
C_START = "#2CA02C"
C_DASH = "#999999"
C_HUMAN = "#AEC7E8"


def load_scene(spec_path):
    taskspec = load_taskspec_from_json(spec_path)
    start = np.asarray(taskspec.phases[0]["start"], dtype=float)
    goal = np.asarray(taskspec.phases[-1]["end"], dtype=float)
    horizon = float(sum(float(p["duration"]) for p in taskspec.phases))
    carry_end = float(taskspec.phases[0]["duration"])

    obs_clause = next((cl for cl in taskspec.clauses if cl.predicate == "ObstacleAvoidance"), None)
    if obs_clause is None:
        raise ValueError("Exp3a spec must contain an ObstacleAvoidance clause.")

    obstacle = np.asarray(obs_clause.parameters["obstacle_position"], dtype=float)
    obs_rad = float(obs_clause.parameters["safe_radius"])
    geometry = str(obs_clause.parameters.get("geometry", "sphere"))

    return {
        "taskspec": taskspec,
        "start": start,
        "goal": goal,
        "horizon": horizon,
        "carry_end": carry_end,
        "obstacle": obstacle,
        "obs_rad": obs_rad,
        "geometry": geometry,
        "human_prox_rad": 0.12,
        "human_ramp_rad": 0.36,
        "obs_hx": 0.08,
        "obs_hy": 0.08,
    }


def load_csv_trace(csv_path, horizon):
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise ValueError(f"No rows found in {csv_path}")
    if data.ndim == 0:
        data = data.reshape(1)

    required = {"x", "y", "z"}
    missing = sorted(required.difference(data.dtype.names))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

    pos = np.column_stack([data["x"], data["y"], data["z"]])
    T = pos.shape[0]
    t = np.linspace(0.0, horizon, T)
    dt = horizon / max(T - 1, 1)

    vel = np.zeros_like(pos)
    if T >= 2:
        vel[0] = (pos[1] - pos[0]) / dt
        vel[-1] = (pos[-1] - pos[-2]) / dt
        if T > 2:
            vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)

    trace = SimpleNamespace(
        time=t,
        position=pos,
        velocity=vel,
        gains={"K": None, "D": None},
        obstacle_debug=None,
    )
    return trace


def maybe_reconstruct_from_checkpoint(scene, checkpoint_path):
    if not checkpoint_path:
        return None, "No checkpoint provided; plotting saved trajectory only."

    ckpt = np.load(checkpoint_path)
    if "best_theta" not in ckpt:
        return None, f"{checkpoint_path} has no `best_theta`; plotting saved trajectory only."

    policy = MultiPhaseCertifiedPolicy(scene["taskspec"].phases, K0=300.0, D0=30.0)
    policy.setup_hard_obstacles_from_taskspec(scene["taskspec"])

    theta = ckpt["best_theta"]
    theta_dim = policy.parameter_dimension()
    if theta.shape[0] != theta_dim:
        return None, (
            f"Checkpoint theta_dim mismatch: saved={theta.shape[0]}, expected={theta_dim}. "
            "Plotting saved trajectory only."
        )

    trace_deformed = policy.rollout(theta)
    projector_backup = policy._projector
    policy._projector = ObstacleProjector([])
    trace_nominal = policy.rollout(theta)
    policy._projector = projector_backup
    return {"nominal": trace_nominal, "deformed": trace_deformed}, "Checkpoint reconstruction enabled."


def active_segments_from_trace(trace, obstacle, obs_rad, geometry, support_pad=0.08):
    pos = trace.position
    if geometry == "cylinder_infinite":
        d = np.linalg.norm(pos[:, :2] - obstacle[:2], axis=1)
    else:
        d = np.linalg.norm(pos - obstacle, axis=1)
    active = d < (obs_rad + support_pad)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [chunk for chunk in np.split(idx, splits) if chunk.size > 0]


def plot_topdown(scene, trace_actual, prefix, trace_nominal=None, trace_deformed=None):
    start = scene["start"]
    goal = scene["goal"]
    obstacle = scene["obstacle"]
    obs_rad = scene["obs_rad"]
    carry_end = scene["carry_end"]
    ramp_rad = scene["human_ramp_rad"]
    prox_rad = scene["human_prox_rad"]
    side_half = max(scene["obs_hx"], scene["obs_hy"])

    pos = trace_actual.position
    t = trace_actual.time
    ip = np.searchsorted(t, carry_end)

    fig, ax = plt.subplots(figsize=(9.2, 6.4))

    obs_square = mpatches.Rectangle(
        (obstacle[0] - side_half, obstacle[1] - side_half),
        2 * side_half,
        2 * side_half,
        facecolor=C_OBS,
        edgecolor="#666666",
        linewidth=1.0,
        alpha=0.35,
        zorder=2,
        label="Obstacle footprint",
    )
    obs_circle = plt.Circle(
        (obstacle[0], obstacle[1]),
        obs_rad,
        fill=False,
        color="#666666",
        linestyle="--",
        linewidth=1.3,
        alpha=0.9,
        zorder=3,
        label=f"Obstacle model (r={obs_rad:.2f} m)",
    )
    ax.add_patch(obs_square)
    ax.add_patch(obs_circle)

    ax.add_patch(plt.Circle((goal[0], goal[1]), ramp_rad, color=C_HUMAN, alpha=0.08, zorder=1))
    ax.add_patch(plt.Circle((goal[0], goal[1]), prox_rad, color=C_HUMAN, alpha=0.20, zorder=1))

    ax.plot([start[0], goal[0]], [start[1], goal[1]],
            "--", color=C_DASH, lw=1.5, alpha=0.55, zorder=3, label="Shortest path")

    if trace_nominal is not None:
        pnom = trace_nominal.position
        ax.plot(pnom[:, 0], pnom[:, 1],
                ls="--", lw=1.2, color=C_NOM, alpha=0.75,
                zorder=3, label="Nominal (pre-deform)")

    ax.plot(pos[:ip + 1, 0], pos[:ip + 1, 1],
            color=C_ACTUAL, lw=2.2, zorder=5, label="Saved trajectory")
    if pos.shape[0] > ip:
        ax.plot(pos[ip:, 0], pos[ip:, 1], color="#C44E52", lw=2.0, zorder=5, label="Pour")

    active_segments = active_segments_from_trace(trace_actual, obstacle, obs_rad, scene["geometry"])
    for i, seg in enumerate(active_segments):
        ax.plot(pos[seg, 0], pos[seg, 1],
                color=C_ACTIVE, lw=2.9, alpha=0.92, zorder=7,
                label="Near-obstacle segment" if i == 0 else None)

    if trace_deformed is not None and getattr(trace_deformed, "obstacle_debug", None):
        dbg = trace_deformed.obstacle_debug
        colliding_idx = np.asarray(dbg.get("collision_indices", []), dtype=int)
        projected = np.asarray(dbg.get("projected_position"), dtype=float)
        valid = np.all(np.isfinite(projected), axis=1) if projected.size else np.zeros(0, dtype=bool)

        if trace_nominal is not None and colliding_idx.size > 0:
            colliding_nom = trace_nominal.position[colliding_idx]
            step = max(1, len(colliding_nom) // 14)
            ax.scatter(colliding_nom[::step, 0], colliding_nom[::step, 1],
                       s=14, facecolors="white", edgecolors="#333333",
                       linewidths=0.7, alpha=0.75, zorder=6, label="Colliding samples")

            valid_idx = colliding_idx[valid[colliding_idx]]
            if valid_idx.size > 0:
                step = max(1, len(valid_idx) // 8)
                base = trace_nominal.position[valid_idx]
                delta = projected[valid_idx] - base
                ax.quiver(base[::step, 0], base[::step, 1],
                          delta[::step, 0], delta[::step, 1],
                          angles="xy", scale_units="xy", scale=1.0,
                          width=0.0016, color="#666666", alpha=0.55,
                          zorder=6, label="Projection correction")

        if projected.size > 0 and valid.any():
            proj_idx = np.flatnonzero(valid)
            step = max(1, len(proj_idx) // 10)
            pts = projected[proj_idx[::step]]
            ax.scatter(pts[:, 0], pts[:, 1], s=16, marker="x", color="#222222",
                       linewidths=0.7, alpha=0.65, zorder=7, label="Radial projected points")

    ax.scatter(start[0], start[1], s=85, c=C_START, edgecolors="black", linewidth=0.7, zorder=10, label="Start")
    ax.scatter(goal[0], goal[1], s=75, c=C_GOAL, marker="D", edgecolors="black", linewidth=0.7, zorder=10, label="Goal")
    ax.text(obstacle[0], obstacle[1] + side_half + 0.02, "Obstacle", fontsize=8, ha="center", color="#555555")
    ax.text(start[0] + 0.01, start[1] - 0.035, "Start", fontsize=8, fontweight="bold", color=C_START)
    ax.text(goal[0] + 0.01, goal[1] + 0.015, "Human\n(goal)", fontsize=8, fontweight="bold", color=C_GOAL)

    ax.set_xlabel("X — forward/depth (m)")
    ax.set_ylabel("Y — lateral (m)")
    ax.set_title("Exp 3a — trajectory from saved data")
    ax.set_aspect("equal")

    margin = 0.08
    xlo = min(start[0], goal[0], obstacle[0] - side_half) - margin
    xhi = max(start[0], goal[0], obstacle[0] + side_half) + margin
    ylo = min(start[1], goal[1], obstacle[1] - side_half) - margin
    yhi = max(start[1], goal[1], obstacle[1] + side_half) + margin
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              framealpha=0.92, edgecolor="lightgrey", fancybox=False)

    axins = inset_axes(ax, width="32%", height="32%", loc="lower left",
                       bbox_to_anchor=(0.05, 0.05, 1, 1),
                       bbox_transform=ax.transAxes, borderpad=0.8)
    axins.add_patch(plt.Circle((obstacle[0], obstacle[1]), obs_rad,
                               fill=False, color="#666666", linestyle="--",
                               linewidth=1.0, alpha=0.9))
    if trace_nominal is not None:
        pnom = trace_nominal.position
        axins.plot(pnom[:, 0], pnom[:, 1], ls="--", lw=1.0, color=C_NOM, alpha=0.75)
    axins.plot(pos[:, 0], pos[:, 1], color=C_ACTUAL, lw=1.5)
    for seg in active_segments:
        axins.plot(pos[seg, 0], pos[seg, 1], color=C_ACTIVE, lw=2.2)
    axins.set_xlim(obstacle[0] - obs_rad - 0.08, obstacle[0] + obs_rad + 0.08)
    axins.set_ylim(obstacle[1] - obs_rad - 0.08, obstacle[1] + obs_rad + 0.08)
    axins.set_aspect("equal")
    axins.grid(True, alpha=0.15)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#999999", alpha=0.6)

    plt.tight_layout()
    out = f"{prefix}_topdown.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


def plot_clearance(scene, trace_actual, prefix):
    t = trace_actual.time
    pos = trace_actual.position
    obstacle = scene["obstacle"]
    obs_rad = scene["obs_rad"]

    if scene["geometry"] == "cylinder_infinite":
        d = np.linalg.norm(pos[:, :2] - obstacle[:2], axis=1)
    else:
        d = np.linalg.norm(pos - obstacle, axis=1)
    clearance = d - obs_rad

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(t, clearance, color=C_ACTUAL, lw=2.1, label="Clearance to obstacle")
    ax.axhline(0.0, color="#C44E52", ls="--", lw=1.2, alpha=0.8, label="Obstacle boundary")
    ax.axvspan(0.0, scene["carry_end"], alpha=0.03, color=C_ACTUAL)
    ax.axvspan(scene["carry_end"], scene["horizon"], alpha=0.03, color="#C44E52")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Clearance (m)")
    ax.set_title("Exp 3a — saved trajectory obstacle clearance")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5, loc="best", framealpha=0.9, edgecolor="lightgrey", fancybox=False)
    plt.tight_layout()
    out = f"{prefix}_clearance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Standalone Exp 3a plotting from saved data.")
    parser.add_argument("--csv", default="exp3a_trajectory.csv", help="Saved Exp 3a trajectory CSV.")
    parser.add_argument("--spec", default="spec/exp3a_task.json", help="Exp 3a task spec JSON.")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional checkpoint with compatible `best_theta` for nominal/correction overlay.")
    parser.add_argument("--prefix", default="exp3a_data", help="Output prefix.")
    args = parser.parse_args()

    scene = load_scene(args.spec)
    trace_actual = load_csv_trace(args.csv, scene["horizon"])
    recon, msg = maybe_reconstruct_from_checkpoint(scene, args.checkpoint)
    print(msg)

    trace_nominal = recon["nominal"] if recon is not None else None
    trace_deformed = recon["deformed"] if recon is not None else None

    plot_topdown(scene, trace_actual, args.prefix, trace_nominal=trace_nominal, trace_deformed=trace_deformed)
    plot_clearance(scene, trace_actual, args.prefix)


if __name__ == "__main__":
    main()
