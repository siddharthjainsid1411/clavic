# logic/predicates.py

import numpy as np
from core.cgms.quat_utils import quat_distance, quat_error, quat_normalize


def at_goal_pose(trace, target, tolerance=0.02):
    pos = trace.position
    d = np.linalg.norm(pos - target, axis=1)
    return tolerance - d


def human_comfort_distance(trace, human_position, preferred_distance):
    pos = trace.position
    d = np.linalg.norm(pos - human_position, axis=1)
    return d - preferred_distance


def human_body_exclusion(trace, human_position, body_radius):
    """
    Hard exclusion zone — the physical body of the human.
    Robot must NEVER enter this radius. Use with modality=REQUIRE.
    rho > 0  means safe (outside body)
    rho < 0  means collision (inside body)
    """
    pos = trace.position
    d = np.linalg.norm(pos - human_position, axis=1)
    return d - body_radius


def velocity_limit(trace, vmax):
    if trace.velocity is None:
        raise ValueError("Velocity not available in trace.")
    v_norm = np.linalg.norm(trace.velocity, axis=1)
    return vmax - v_norm


# ------------------------------------------------------------------ #
#                    Scene-2 predicates                               #
# ------------------------------------------------------------------ #

def obstacle_avoidance(trace, obstacle_position, safe_radius):
    """
    Hard exclusion around a static obstacle.
    rho > 0  →  safe (outside radius)
    rho < 0  →  collision
    """
    pos = trace.position
    d = np.linalg.norm(pos - obstacle_position, axis=1)
    return d - safe_radius


def at_waypoint(trace, waypoint, tolerance=0.03):
    """
    Robot is within *tolerance* of a waypoint.
    Same semantics as at_goal_pose but with a distinct name so JSON
    can bind different targets to the two predicates independently.
    """
    pos = trace.position
    d = np.linalg.norm(pos - waypoint, axis=1)
    return tolerance - d


def zero_velocity(trace, speed_threshold=0.05):
    """
    Robot velocity magnitude is below *speed_threshold*.
    rho > 0  →  almost stationary
    rho < 0  →  moving too fast
    """
    if trace.velocity is None:
        raise ValueError("Velocity not available in trace.")
    v_norm = np.linalg.norm(trace.velocity, axis=1)
    return speed_threshold - v_norm


def hold_at_waypoint(trace, waypoint, tolerance=0.03, speed_threshold=0.05):
    """
    Conjunction: robot is near waypoint AND nearly stationary.
    rho = min(at_waypoint, zero_velocity)  — both must be satisfied.
    """
    rho_pos = at_waypoint(trace, waypoint, tolerance)
    rho_vel = zero_velocity(trace, speed_threshold)
    return np.minimum(rho_pos, rho_vel)


def early_completion(trace, target, tolerance=0.05, early_time=1.0):
    """
    Reward for reaching the goal *before* early_time.
    rho > 0  →  goal reached before early_time
    rho < 0  →  goal not reached early enough
    Used in compare_tau_initialization.py to reward fast (tau=0.5s) solutions.
    """
    pos = trace.position
    d   = np.linalg.norm(pos - target, axis=1)
    at_goal = tolerance - d                 # > 0 when inside tolerance
    # Find if goal was reached before early_time
    mask_early = trace.time <= early_time
    if np.any(mask_early):
        return float(np.max(at_goal[mask_early]))
    return float(np.max(at_goal))


# ------------------------------------------------------------------ #
#                 Orientation predicates                              #
# ------------------------------------------------------------------ #

def orientation_at_target(trace, q_target, tolerance_rad=0.1):
    """
    Orientation is within geodesic tolerance of a target quaternion.
    rho > 0  →  within tolerance
    rho < 0  →  outside tolerance

    Parameters
    ----------
    trace        : Trace with orientation (T, 4)
    q_target     : array (4,) target quaternion [w,x,y,z]
    tolerance_rad: float geodesic angle tolerance (radians)
    """
    if trace.orientation is None:
        raise ValueError("Orientation not available in trace.")
    q_target = quat_normalize(np.asarray(q_target, float))
    T = trace.orientation.shape[0]
    rho = np.zeros(T)
    for k in range(T):
        d = quat_distance(trace.orientation[k], q_target)
        rho[k] = tolerance_rad - d
    return rho


def orientation_hold(trace, q_target, tolerance_rad=0.1, omega_max=0.05):
    """
    Conjunction: orientation near target AND angular velocity low.
    rho = min(orientation_at_target, angular_velocity_limit)

    Parameters
    ----------
    trace         : Trace
    q_target      : array (4,) target quaternion
    tolerance_rad : float geodesic tolerance (radians)
    omega_max     : float max angular velocity norm (rad/s)
    """
    rho_ori = orientation_at_target(trace, q_target, tolerance_rad)
    rho_vel = angular_velocity_limit(trace, omega_max)
    return np.minimum(rho_ori, rho_vel)


def orientation_limit(trace, q_ref, max_angle_rad=0.5):
    """
    Orientation stays within max_angle of a reference quaternion.
    rho > 0  →  within angular limit
    rho < 0  →  exceeded

    Parameters
    ----------
    trace          : Trace
    q_ref          : array (4,) reference quaternion
    max_angle_rad  : float max geodesic deviation (radians)
    """
    if trace.orientation is None:
        raise ValueError("Orientation not available in trace.")
    q_ref = quat_normalize(np.asarray(q_ref, float))
    T = trace.orientation.shape[0]
    rho = np.zeros(T)
    for k in range(T):
        d = quat_distance(trace.orientation[k], q_ref)
        rho[k] = max_angle_rad - d
    return rho


def angular_velocity_limit(trace, omega_max=1.0):
    """
    Angular velocity norm below a threshold.
    rho > 0  →  safe
    rho < 0  →  too fast

    Parameters
    ----------
    trace     : Trace
    omega_max : float max angular velocity (rad/s)
    """
    if trace.angular_velocity is None:
        raise ValueError("Angular velocity not available in trace.")
    omega_norm = np.linalg.norm(trace.angular_velocity, axis=1)
    return omega_max - omega_norm


def dont_pour_until_at_goal(trace, q_pour, pour_tolerance_rad=0.3,
                            goal_position=None, goal_tolerance=0.05):
    """
    "Don't pour (reach pour orientation) until you reach the goal position."

    This is an UNTIL predicate: orientation must stay away from pour angle
    UNTIL position reaches the goal.

    Returns two robustness signals (for use with temporal_logic.until):
      rho_phi : "not pouring" — orientation far from pour orientation
      rho_psi : "at goal"     — position within tolerance of goal

    Parameters
    ----------
    trace              : Trace
    q_pour             : array (4,) pour orientation quaternion
    pour_tolerance_rad : float angle within which counts as "pouring"
    goal_position      : array (3,) goal Cartesian position
    goal_tolerance     : float position tolerance (m)
    """
    if trace.orientation is None:
        raise ValueError("Orientation not available in trace.")
    q_pour = quat_normalize(np.asarray(q_pour, float))
    T = trace.orientation.shape[0]

    # "Not pouring" = orientation is far from q_pour
    rho_phi = np.zeros(T)
    for k in range(T):
        d = quat_distance(trace.orientation[k], q_pour)
        rho_phi[k] = d - pour_tolerance_rad  # > 0 when NOT pouring

    # "At goal" = position reached goal
    rho_psi = goal_tolerance - np.linalg.norm(
        trace.position - np.asarray(goal_position, float), axis=1
    )

    return rho_phi, rho_psi