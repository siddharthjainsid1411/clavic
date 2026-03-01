"""
Scene 2 — Multi-phase task:
    Phase 1: start → waypoint  (avoid obstacle)
    Phase 2: hold at waypoint  (zero velocity, 2 sec)
    Phase 3: waypoint → goal

Uses MultiPhaseCertifiedPolicy (3 chained DMPs) + temporal-logic
clauses with time windows (always_during, eventually_during).
"""

import numpy as np
import matplotlib.pyplot as plt

from spec.compiler import Compiler
from spec.json_parser import load_taskspec_from_json
from core.multi_phase_policy import MultiPhaseCertifiedPolicy
from optimization.optimizer import PI2
from experiment_checkpoint_warmstart import save_checkpoint

from logic.predicates import (
    at_goal_pose, at_waypoint, hold_at_waypoint,
    obstacle_avoidance, velocity_limit,
    human_comfort_distance, human_body_exclusion,
)


# ----------------------------
# Predicate Registry
# ----------------------------
def build_predicate_registry():
    return {
        "AtGoalPose":           at_goal_pose,
        "AtWaypoint":           at_waypoint,
        "HoldAtWaypoint":       hold_at_waypoint,
        "ObstacleAvoidance":    obstacle_avoidance,
        "VelocityLimit":        velocity_limit,
        # keep scene-1 predicates available for mixed specs
        "HumanComfortDistance":  human_comfort_distance,
        "HumanBodyExclusion":   human_body_exclusion,
    }


# ----------------------------
# Main
# ----------------------------
def main():

    taskspec = load_taskspec_from_json("spec/scene2_task.json")

    # ---- Build multi-phase policy from JSON phases ----
    assert taskspec.phases is not None, "scene2_task.json must contain 'phases'"
    policy = MultiPhaseCertifiedPolicy(taskspec.phases)
    theta_dim = policy.parameter_dimension()
    print(f"Multi-phase policy: {len(taskspec.phases)} phases, "
          f"theta_dim = {theta_dim}")

    # ---- Compiler ----
    predicate_registry = build_predicate_registry()
    compiler = Compiler(predicate_registry)
    objective_fn = compiler.compile(taskspec)

    # ---- Nominal trace (theta=0) diagnostics ----
    trace0 = policy.rollout(np.zeros(theta_dim))
    print(f"\nNominal trace: {trace0.time[0]:.2f}s → {trace0.time[-1]:.2f}s  "
          f"({len(trace0.time)} steps)")
    print(f"  start pos : {trace0.position[0]}")
    print(f"  end pos   : {trace0.position[-1]}")
    cost0 = objective_fn(trace0)
    print(f"  nominal cost: {cost0:.4f}\n")

    # ---- PI2 Optimizer ----
    theta_init = np.zeros(theta_dim)
    sigma_init = policy.structured_sigma()

    # Reduce exploration noise for the hold phase — its nominal is already
    # nearly correct (start == end → min-jerk stays put).  Large noise only
    # causes the held position to drift and violate HoldAtWaypoint.
    hold_off   = policy.offsets[1]
    hold_dim   = policy.theta_dims[1]
    sigma_init[hold_off:hold_off + hold_dim] *= 0.2   # 5× smaller sigma

    pi2 = PI2(
        theta=theta_init,
        sigma=sigma_init,
        lam=1.0,              # less greedy — prevents mean divergence
        decay=0.99,           # moderate sigma shrinkage
    )

    N_SAMPLES  = 15           # balance: coverage vs speed
    N_UPDATES  = 100          # sufficient for convergence

    best_cost  = float("inf")
    best_theta = theta_init.copy()

    print("Starting Optimization ...")
    for update_idx in range(N_UPDATES):
        samples = pi2.sample(N_SAMPLES)
        costs = np.array([
            objective_fn(policy.rollout(samples[i]))
            for i in range(N_SAMPLES)
        ])

        current_mean, new_sigma, weights = pi2.update(samples, costs)

        if costs.min() < best_cost:
            best_cost  = costs.min()
            best_theta = samples[np.argmin(costs)].copy()

        if (update_idx + 1) % 10 == 0 or update_idx == 0:
            print(f"  Update {update_idx+1:03d} | "
                  f"Min: {costs.min():.4f} | "
                  f"Mean: {costs.mean():.4f} | "
                  f"BestSoFar: {best_cost:.4f}")

    print("Optimization Complete.\n")

    # ---- Evaluate best solution ----
    trace_final = policy.rollout(best_theta)

    # Save checkpoint
    save_checkpoint(best_theta, taskspec.horizon_sec, best_cost, trace_final)

    # Extract key bindings for analysis
    goal     = np.array([0.05, 0.72, 0.11])
    waypoint = np.array([0.30, 0.55, 0.11])
    obstacle = np.array([0.40, 0.30, 0.11])

    rho_goal = at_goal_pose(trace_final, goal, 0.05)
    goal_hit = trace_final.time[rho_goal > 0]
    print(f"Goal reached: {'Yes at t=' + f'{goal_hit[0]:.2f}s' if len(goal_hit) else 'NO'}")

    rho_wp = at_waypoint(trace_final, waypoint, 0.05)
    wp_hit = trace_final.time[rho_wp > 0]
    print(f"Waypoint reached: {'Yes at t=' + f'{wp_hit[0]:.2f}s' if len(wp_hit) else 'NO'}")

    obs_dist = np.linalg.norm(trace_final.position - obstacle, axis=1)
    print(f"Min obstacle clearance: {(np.min(obs_dist) - 0.08)*100:.1f} cm "
          f"(body={np.min(obs_dist)*100:.1f} cm, radius=8 cm)")

    speed = np.linalg.norm(trace_final.velocity, axis=1)
    mask_hold = (trace_final.time >= 2.0) & (trace_final.time <= 4.0)
    print(f"Max speed during hold (2-4s): {np.max(speed[mask_hold]):.4f} m/s")

    # ---- Plotting ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Workspace XY
    ax = axes[0, 0]
    pos = trace_final.position
    ax.plot(pos[:, 0], pos[:, 1], 'r-', linewidth=2, label="Learned")
    ax.scatter(*pos[0, :2], c='green', s=100, zorder=5, label="Start")
    ax.scatter(*goal[:2], c='black', s=100, zorder=5, label="Goal")
    ax.scatter(*waypoint[:2], c='blue', s=100, zorder=5, marker='s', label="Waypoint")
    ax.scatter(*obstacle[:2], c='red', s=100, zorder=5, marker='x', label="Obstacle")
    circ_obs = plt.Circle(obstacle[:2], 0.08, color='red', fill=True,
                          alpha=0.2, label="Obstacle zone (r=0.08)")
    ax.add_patch(circ_obs)
    circ_wp = plt.Circle(waypoint[:2], 0.05, color='blue', fill=False,
                         linestyle=':', linewidth=2, label="Waypoint tol")
    ax.add_patch(circ_wp)
    # Phase boundaries
    for ph_t in [2.0, 4.0]:
        idx = np.searchsorted(trace_final.time, ph_t)
        if idx < len(pos):
            ax.scatter(pos[idx, 0], pos[idx, 1], c='orange', s=60, zorder=6,
                       marker='D')
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Workspace (XY)"); ax.legend(fontsize=7); ax.axis('equal')
    ax.grid(True)

    # 2. Speed vs time
    ax = axes[0, 1]
    ax.plot(trace_final.time, speed, 'b-')
    ax.axvspan(2.0, 4.0, alpha=0.1, color='green', label='Hold phase')
    ax.axhline(0.05, color='gray', linestyle='--', label='Speed threshold')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed vs Time"); ax.legend(); ax.grid(True)

    # 3. Distance to waypoint vs time
    ax = axes[1, 0]
    d_wp = np.linalg.norm(pos - waypoint, axis=1)
    d_goal = np.linalg.norm(pos - goal, axis=1)
    d_obs = np.linalg.norm(pos - obstacle, axis=1)
    ax.plot(trace_final.time, d_wp, 'b-', label="dist(waypoint)")
    ax.plot(trace_final.time, d_goal, 'k--', label="dist(goal)")
    ax.plot(trace_final.time, d_obs, 'r:', label="dist(obstacle)")
    ax.axhline(0.05, color='blue', linestyle=':', alpha=0.5, label="wp tol")
    ax.axhline(0.08, color='red', linestyle=':', alpha=0.5, label="obs radius")
    ax.axvspan(2.0, 4.0, alpha=0.1, color='green', label='Hold phase')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
    ax.set_title("Distances vs Time"); ax.legend(fontsize=7); ax.grid(True)

    # 4. Stiffness
    ax = axes[1, 1]
    K_trace = trace_final.gains["K"]
    trK = [np.trace(K) for K in K_trace]
    ax.plot(trace_final.time, trK, 'm-')
    ax.axvspan(2.0, 4.0, alpha=0.1, color='green', label='Hold phase')
    ax.set_xlabel("Time (s)"); ax.set_ylabel("tr(K) (N/m)")
    ax.set_title("Stiffness Evolution"); ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig("scene2_result.png", dpi=300)
    print("\nSaved scene2_result.png")
    plt.show()


if __name__ == "__main__":
    main()
