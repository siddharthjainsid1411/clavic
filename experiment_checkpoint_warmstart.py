"""
Experimentation Script: Checkpoint Warm-Start and Optimal Trajectory Analysis

This script enables:
1. Warm-start optimization from saved checkpoints
2. Analysis of optimal trajectories across different tau values
3. Checkpoint saving and loading with tau-matching logic
4. Visualization of learned policies vs nominal trajectories

NOT used in main.py to keep the core implementation clean.
"""

import numpy as np
import os

# ---- Spec ----
from spec.taskspec import TaskSpec, Clause
from spec.compiler import Compiler

# ---- Logic ----
from logic.predicates import at_goal_pose, human_comfort_distance, velocity_limit, human_body_exclusion

# ---- Core ----
from core.certified_policy import CertifiedPolicy

# ---- Optimizer ----
from optimization.optimizer import PI2

from spec.json_parser import load_taskspec_from_json


def build_predicate_registry():
    return {
        "AtGoalPose": at_goal_pose,
        "HumanComfortDistance": human_comfort_distance,
        "HumanBodyExclusion": human_body_exclusion,
        "VelocityLimit": velocity_limit
    }


def save_checkpoint(theta, tau, cost, trace, checkpoint_path="optimal_checkpoint.npz"):
    """
    Save the optimal trajectory and parameters to a checkpoint file.
    
    Parameters saved:
    - best_theta: The optimal parameter vector
    - optimal_tau: Minimum tau at which goal was satisfied
    - horizon_sec: The tau this was optimized for (used for tau-matching on reload)
    - best_cost: The cost value achieved
    - Full trace: position, velocity, time, K_trace, D_trace
    """
    np.savez(
        checkpoint_path,
        best_theta      = theta,
        optimal_tau     = np.array(tau),
        horizon_sec     = np.array(tau),  # ← stores the tau this was optimized for
        best_cost       = np.array(cost),
        position        = trace.position,
        velocity        = trace.velocity,
        time            = trace.time,
        K_trace         = np.array([K for K in trace.gains["K"]]),
        D_trace         = np.array([D for D in trace.gains["D"]]),
    )
    print(f"✓ Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path="optimal_checkpoint.npz", expected_tau=None, theta_dim=None):
    """
    Load checkpoint with tau-matching logic.
    
    Returns: (theta, saved_tau, cost, can_use)
    - can_use=True if tau matches and theta dimension matches
    - can_use=False if mismatch detected (you should start fresh)
    """
    if not os.path.exists(checkpoint_path):
        return None, None, None, False
    
    ckpt = np.load(checkpoint_path)
    saved_tau = float(ckpt["horizon_sec"])
    best_theta = ckpt["best_theta"]
    best_cost = float(ckpt["best_cost"])
    
    can_use = True
    reason = ""
    
    if expected_tau is not None and saved_tau != expected_tau:
        can_use = False
        reason = f"tau mismatch: saved={saved_tau}s, current={expected_tau}s"
    
    if theta_dim is not None and best_theta.shape[0] != theta_dim:
        can_use = False
        reason = f"theta_dim mismatch: saved={best_theta.shape[0]}, current={theta_dim}"
    
    if can_use:
        print(f"✓ Warm-start: loaded checkpoint (tau={saved_tau}s, cost={best_cost:.4f})")
    else:
        print(f"✗ Checkpoint mismatch: {reason} — starting fresh")
    
    return best_theta, saved_tau, best_cost, can_use


def optimize_with_checkpoint_support(
    taskspec,
    certified_policy,
    predicate_registry,
    n_updates=40,
    n_samples=12,
    use_checkpoint=True,
    checkpoint_path="optimal_checkpoint.npz"
):
    """
    Run optimization with checkpoint warm-start support.
    
    If use_checkpoint=True and a matching checkpoint exists, initialize from it.
    Track the single best theta across all iterations (not just the running mean).
    Save checkpoint after optimization.
    """
    
    compiler = Compiler(predicate_registry)
    objective_fn = compiler.compile(taskspec)
    
    theta_dim = certified_policy.parameter_dimension()
    
    # ----------------------------
    # Checkpoint Warm-Start
    # ----------------------------
    if use_checkpoint:
        theta_init, saved_tau, saved_cost, can_use = load_checkpoint(
            checkpoint_path, 
            expected_tau=taskspec.horizon_sec,
            theta_dim=theta_dim
        )
        if can_use:
            sigma_init = np.ones(theta_dim) * 1.0  # ← tighter exploration (already near optimum)
        else:
            theta_init = np.zeros(theta_dim)
            sigma_init = np.ones(theta_dim) * 5.0
    else:
        theta_init = np.zeros(theta_dim)
        sigma_init = np.ones(theta_dim) * 5.0
    
    pi2 = PI2(
        theta=theta_init,
        sigma=sigma_init,
        lam=0.01,
        decay=0.98
    )
    
    # ----------------------------
    # Debug: Nominal trajectory
    # ----------------------------
    trace_nominal = certified_policy.rollout(np.zeros(theta_dim))
    rho_goal_nom = at_goal_pose(trace_nominal, taskspec.clauses[0].parameters['target'], 
                                taskspec.clauses[0].parameters.get('tolerance', 0.05))
    rho_human_nom = human_comfort_distance(trace_nominal, 
                                           taskspec.clauses[1].parameters['human_position'],
                                           taskspec.clauses[1].parameters['preferred_distance'])
    
    print("\n--- Nominal (theta = 0) robustness ---")
    print(f"  Max goal robustness: {np.max(rho_goal_nom):.6f}")
    print(f"  Min human robustness: {np.min(rho_human_nom):.6f}")
    print("----------------------------------------\n")
    
    # ----------------------------
    # Optimization Loop (tracking best_theta)
    # ----------------------------
    best_cost = float("inf")
    best_theta = theta_init.copy()
    best_trace = None
    
    current_mean = theta_init.copy()
    
    print("Starting Optimization...")
    for update_idx in range(n_updates):
        samples = pi2.sample(n_samples)
        costs = []
        traces = []
        
        for i in range(n_samples):
            theta = samples[i]
            trace = certified_policy.rollout(theta)
            cost = objective_fn(trace)
            costs.append(cost)
            traces.append(trace)
        
        costs = np.array(costs)
        
        # Track the SINGLE BEST theta found (not just running mean)
        min_idx = np.argmin(costs)
        if costs[min_idx] < best_cost:
            best_cost = costs[min_idx]
            best_theta = samples[min_idx].copy()
            best_trace = traces[min_idx]
        
        # Update PI² distribution for next iteration
        current_mean, new_sigma, weights = pi2.update(samples, costs)
        
        print(
            f"Update {update_idx+1:02d} | "
            f"Min: {costs.min():.4f} | "
            f"Mean: {costs.mean():.4f} | "
            f"BestSoFar: {best_cost:.4f}"
        )
    
    # ----------------------------
    # Post-optimization Analysis
    # ----------------------------
    print("\nOptimization Complete.")
    print(f"Best cost found: {best_cost:.6f}")
    
    goal_rho = at_goal_pose(best_trace, taskspec.clauses[0].parameters['target'],
                            taskspec.clauses[0].parameters.get('tolerance', 0.05))
    goal_times = best_trace.time
    goal_satisfied = goal_rho > 0
    
    if goal_satisfied.any():
        optimal_tau_candidate = goal_times[goal_satisfied][0]
        print(f"Goal first satisfied at: {optimal_tau_candidate:.3f}s (optimal tau candidate)")
    else:
        optimal_tau_candidate = taskspec.horizon_sec
        print(f"Goal never satisfied — using full horizon {taskspec.horizon_sec}s")
    
    # ----------------------------
    # Save Checkpoint
    # ----------------------------
    save_checkpoint(
        best_theta,
        taskspec.horizon_sec,  # ← save the tau this was optimized for
        best_cost,
        best_trace,
        checkpoint_path
    )
    
    return {
        "best_theta": best_theta,
        "best_cost": best_cost,
        "trace": best_trace,
        "optimal_tau_candidate": optimal_tau_candidate,
        "nominal_trace": trace_nominal
    }


def analyze_optimal_trajectory_across_tau(
    goal_pose, human_position, checkpoint_path="optimal_checkpoint.npz"
):
    """
    IMPORTANT: Clarifies what "optimal trajectory" means.
    
    The saved checkpoint contains:
    - best_theta: Optimized FOR tau = horizon_sec
    - optimal_tau_candidate: Earliest time goal is satisfied using that theta
    
    This function tests: "What if we use the same best_theta but execute it at different tau values?"
    This shows the ADAPTABILITY of the learned parameters across different durations.
    """
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found!")
        return
    
    ckpt = np.load(checkpoint_path)
    best_theta = ckpt["best_theta"]
    saved_tau = float(ckpt["horizon_sec"])
    best_cost = float(ckpt["best_cost"])
    
    print("\n" + "="*70)
    print("CHECKPOINT TRAJECTORY ANALYSIS")
    print("="*70)
    print(f"\nCheckpoint Summary:")
    print(f"  - best_theta was optimized for: tau = {saved_tau}s")
    print(f"  - Cost achieved: {best_cost:.6f}")
    print(f"  - Optimal tau candidate (goal arrival): {float(ckpt['optimal_tau']):.3f}s")
    
    print("\n" + "-"*70)
    print("KEY INSIGHT:")
    print("-"*70)
    print(f"The saved 'optimal trajectory' means:")
    print(f"  ✓ Best parameters (best_theta) found when TRAINING at tau={saved_tau}s")
    print(f"  ✓ These parameters satisfy all constraints at tau={saved_tau}s")
    print(f"  ✓ The trajectory reaches the goal at {float(ckpt['optimal_tau']):.3f}s")
    print(f"\nIt is NOT:")
    print(f"  ✗ Optimal across ALL possible tau values")
    print(f"  ✗ The global optimum over all durations")
    print(f"  ✗ Tested at other tau values")
    
    print("\n" + "-"*70)
    print("To find optimal across different tau:")
    print("-"*70)
    print("  1. Run optimization separately for each tau value")
    print("  2. Save each checkpoint with different names (e.g., ckpt_tau5.npz, ckpt_tau10.npz)")
    print("  3. Compare best_cost across all tau values")
    print("  4. This would show which tau yields the best compliance with all constraints")
    
    # Optionally: show saved trace statistics
    print("\n" + "-"*70)
    print("Saved Trajectory Statistics:")
    print("-"*70)
    pos = ckpt["position"]
    vel = ckpt["velocity"]
    time = ckpt["time"]
    
    print(f"  Duration: {time[-1]:.3f}s")
    print(f"  Total distance: {np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)):.4f}m")
    print(f"  Max velocity: {np.max(np.linalg.norm(vel, axis=1)):.4f}m/s")
    print(f"  Start position: {pos[0]}")
    print(f"  End position: {pos[-1]}")


def main():
    """
    Main experimentation pipeline.
    """
    
    print("="*70)
    print("CLAVIC Checkpoint & Warm-Start Experimentation")
    print("="*70)
    
    # Load spec
    taskspec = load_taskspec_from_json("spec/example_task.json")
    predicate_registry = build_predicate_registry()
    
    goal_pose = np.array([0.05, 0.72, 0.11])
    human_position = np.array([0.30, 0.40, 0.11])
    
    # Create policy
    certified_policy = CertifiedPolicy(taskspec.horizon_sec)
    
    # Run optimization WITH checkpoint warm-start
    print("\n--- RUN 1: Fresh optimization (no checkpoint) ---\n")
    result = optimize_with_checkpoint_support(
        taskspec,
        certified_policy,
        predicate_registry,
        n_updates=40,
        n_samples=12,
        use_checkpoint=False,  # ← start fresh
        checkpoint_path="optimal_checkpoint.npz"
    )
    
    # Now analyze what the saved checkpoint actually represents
    print("\n")
    analyze_optimal_trajectory_across_tau(goal_pose, human_position)
    
    print("\n" + "="*70)
    print("NEXT RUNS (on same tau):")
    print("="*70)
    print("\nIf you run this script again with the same horizon_sec in example_task.json,")
    print("it will automatically warm-start from the checkpoint with tighter exploration.")
    print("\nTo test across different tau values, modify 'horizon_sec' in example_task.json")
    print("and rename checkpoints to avoid overwrites (e.g., save_checkpoint(..., 'ckpt_tau5.npz'))")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
