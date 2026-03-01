# logic/predicates.py

import numpy as np


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