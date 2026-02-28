import numpy as np

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
from experiment_checkpoint_warmstart import save_checkpoint



# ----------------------------
# Predicate Registry
# ----------------------------
def build_predicate_registry():
    return {
        "AtGoalPose": at_goal_pose,
        "HumanComfortDistance": human_comfort_distance,
        "HumanBodyExclusion": human_body_exclusion,
        "VelocityLimit": velocity_limit,
    }


# ----------------------------
# Main Execution
# ----------------------------
def main():

    # ---- Scene Setup (MATCHES CGMS BACKBONE) ----
    goal_pose = np.array([0.05, 0.72, 0.11])   # same as CertifiedPolicy
    human_position = np.array([0.30, 0.40, 0.11])  # inside reachable workspace

    #taskspec = build_taskspec(goal_pose, human_position)
    taskspec = load_taskspec_from_json("spec/example_task.json")

    predicate_registry = build_predicate_registry()

    compiler = Compiler(predicate_registry)
    objective_fn = compiler.compile(taskspec)

    certified_policy = CertifiedPolicy(taskspec.horizon_sec)

    theta_dim = certified_policy.parameter_dimension()

    theta_init = np.zeros(theta_dim)
    sigma_init = certified_policy.structured_sigma()  # uniform σ=5.0 for all groups

    pi2 = PI2(
        theta=theta_init,
        sigma=sigma_init,
        lam=0.01,
        decay=0.98
    )

    # ----------------------------
    # DEBUG: Check nominal trajectory robustness
    # ----------------------------
    trace = certified_policy.rollout(np.zeros(theta_dim))

    rho_goal = at_goal_pose(trace, goal_pose, 0.05)
    rho_human = human_comfort_distance(trace, human_position, 0.15)

    print("\n--- Nominal (theta = 0) robustness ---")
    print("Nominal max goal robustness:", np.max(rho_goal))
    print("Nominal min human robustness:", np.min(rho_human))
    print("----------------------------------------\n")

    print("Starting Optimization...")

    N_SAMPLES = 12
    N_UPDATES = 100

    best_cost = float("inf")

    current_mean = theta_init.copy()
    best_theta   = theta_init.copy()   # track the single best sample ever seen

    # ----------------------------
    # Optimization Loop
    # ----------------------------
    for update_idx in range(N_UPDATES):

        samples = pi2.sample(N_SAMPLES)
        costs = []

        for i in range(N_SAMPLES):
            theta = samples[i]
            trace = certified_policy.rollout(theta)
            cost = objective_fn(trace)
            costs.append(cost)

        costs = np.array(costs)

        # IMPORTANT: store updated mean
        current_mean, new_sigma, weights = pi2.update(samples, costs)

        # Track best sample — NOT the mean (mean can have drifted SK weights)
        if costs.min() < best_cost:
            best_cost  = costs.min()
            best_theta = samples[np.argmin(costs)].copy()

        print(
            f"Update {update_idx+1:02d} | "
            f"Min: {costs.min():.4f} | "
            f"Mean: {costs.mean():.4f} | "
            f"BestSoFar: {best_cost:.4f}"
        )

    print("Optimization Complete.")
    # Use best_theta (best single sample ever) not current_mean (drifted distribution mean)
    trace_final = certified_policy.rollout(best_theta)

    # Save checkpoint so compare_tau_initialization.py picks up this result
    save_checkpoint(best_theta, taskspec.horizon_sec, best_cost, trace_final)

    rho_human_final = human_comfort_distance(trace_final, human_position, 0.15)
    print("Final min human robustness:", np.min(rho_human_final))

    print("Learned tau:", certified_policy.dmp.tau)

    goal_rho_trace = at_goal_pose(trace_final, goal_pose, 0.05)
    goal_times = trace_final.time
    goal_satisfied_indices = goal_rho_trace > 0

    if goal_satisfied_indices.any():
        first_hit_time = goal_times[goal_satisfied_indices][0]
        print("Goal first satisfied at time:", first_hit_time)
    else:
        print("Goal never satisfied")

    # ----------------------------
    # Visualize Nominal vs Learned
    # ----------------------------
    import matplotlib.pyplot as plt

    trace_nominal = certified_policy.rollout(np.zeros(theta_dim))
    trace_learned = certified_policy.rollout(best_theta)   # best sample, not drifted mean

    speed = np.linalg.norm(trace_learned.velocity, axis=1)

    plt.figure()
    plt.plot(trace_learned.time, speed)
    plt.xlabel("Time (s)")
    plt.ylabel("Speed magnitude")
    plt.title("Speed vs Time")
    plt.grid(True)
    plt.savefig("speed.png", dpi=300)
    print("Saved speed.png")

    K_trace = trace_learned.gains["K"]

    stiff_trace = [np.trace(K) for K in K_trace]

    plt.figure()
    plt.plot(trace_learned.time, stiff_trace)
    plt.xlabel("Time (s)")
    plt.ylabel("Trace(K)")
    plt.title("Stiffness Evolution")
    plt.grid(True)
    plt.savefig("stiffness.png", dpi=300)
    print("Saved stiffness.png")

    pos_nom = trace_nominal.position
    pos_learned = trace_learned.position

    plt.figure(figsize=(6,6))

    plt.plot(pos_nom[:,0], pos_nom[:,1], 'b--', label="Nominal")
    plt.plot(pos_learned[:,0], pos_learned[:,1], 'r', label="Learned")

    plt.scatter(pos_nom[0,0], pos_nom[0,1], c='green', s=100, label="Start")
    plt.scatter(goal_pose[0], goal_pose[1], c='black', s=100, label="Goal")
    plt.scatter(human_position[0], human_position[1], c='red', s=100, label="Human", zorder=5)

    # Human comfort distance (soft constraint) — orange dashed ring
    circle_comfort = plt.Circle(
        (human_position[0], human_position[1]),
        0.15,
        color='orange',
        fill=False,
        linestyle=':',
        linewidth=2,
        label="Comfort zone (r=0.15m)"
    )
    plt.gca().add_patch(circle_comfort)

    # Human body exclusion (hard constraint) — red filled circle
    circle_body = plt.Circle(
        (human_position[0], human_position[1]),
        0.06,
        color='red',
        fill=True,
        alpha=0.3,
        linewidth=2,
        label="Body exclusion (r=0.06m)"
    )
    plt.gca().add_patch(circle_body)

    plt.legend(fontsize=8, loc='best', framealpha=0.95)
    plt.axis('equal')
    plt.title("Nominal vs Learned Trajectory")
    plt.tight_layout()
    plt.show()
    plt.savefig("traj.png", dpi=300)
    print("Saved traj.png")


if __name__ == "__main__":
    main()