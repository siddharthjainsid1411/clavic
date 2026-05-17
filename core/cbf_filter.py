"""
Runtime Control Barrier Function filters.

The filters here operate on the DMP rollout abstraction, not on full robot
torque dynamics.  For the Cartesian velocity limit we filter acceleration:

    x_dot = v
    v_dot = a

with barrier h(v) = vmax^2 - ||v||^2.  The CBF condition

    h_dot + alpha h >= 0

becomes the linear half-space constraint

    2 v^T a <= alpha (vmax^2 - ||v||^2).

The returned acceleration is the Euclidean projection of the nominal DMP
acceleration onto that half-space, i.e. the minimum modification needed to
satisfy the continuous-time CBF inequality.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class VelocityCBFConfig:
    """Configuration for a Cartesian velocity CBF."""

    vmax: float
    alpha: float = 10.0
    enabled: bool = True
    tolerance: float = 1e-10
    # Optional time windows: list of (t_start, t_end, vmax)
    windows: Optional[List[Tuple[float, float, float]]] = None


@dataclass
class OrientationHOCBFConfig:
    """Configuration for a quaternion cone HOCBF."""

    q_ref: np.ndarray
    max_angle_rad: float
    alpha1: float = 8.0
    alpha2: float = 8.0
    t_start: float = -np.inf
    t_end: float = np.inf
    enabled: bool = True
    tolerance: float = 1e-10
    finite_difference_dt: float = 1e-4


@dataclass
class AngularVelocityCBFConfig:
    """Configuration for an angular velocity CBF."""

    omega_max: float
    alpha: float = 10.0
    t_start: float = -np.inf
    t_end: float = np.inf
    enabled: bool = True
    tolerance: float = 1e-10


def filter_velocity_acceleration(a_nom, v, config, t=None):
    """
    Project nominal acceleration onto the velocity-CBF safe half-space.

    Parameters
    ----------
    a_nom : array-like, shape (3,)
        Nominal Cartesian acceleration from the DMP dynamics.
    v : array-like, shape (3,)
        Current Cartesian velocity.
    config : VelocityCBFConfig or None
        CBF configuration.  If None or disabled, acceleration is unchanged.
    t : float or None
        Global time (seconds). Used to activate time-windowed limits.

    Returns
    -------
    a_safe : ndarray, shape (3,)
        Filtered acceleration.
    diagnostics : dict
        h, cbf_lhs, cbf_rhs, active, correction_norm, speed.
    """
    a_nom = np.asarray(a_nom, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)

    speed_sq = float(np.dot(v, v))
    speed = float(np.sqrt(speed_sq))

    def _active_vmax(cfg, t_val):
        if cfg is None or not cfg.enabled:
            return None
        if cfg.windows:
            if t_val is None:
                return min(v for _, _, v in cfg.windows)
            active = [v for ts, te, v in cfg.windows
                      if ts - 1e-12 <= t_val <= te + 1e-12]
            if not active:
                return None
            return min(active)
        return float(cfg.vmax)

    if config is None or not config.enabled:
        return a_nom.copy(), {
            "h": np.inf,
            "cbf_lhs": -np.inf,
            "cbf_rhs": np.inf,
            "active": False,
            "correction_norm": 0.0,
            "speed": speed,
            "window_active": False,
            "vmax_active": None,
        }

    vmax_active = _active_vmax(config, t)
    if vmax_active is None:
        return a_nom.copy(), {
            "h": np.inf,
            "cbf_lhs": -np.inf,
            "cbf_rhs": np.inf,
            "active": False,
            "correction_norm": 0.0,
            "speed": speed,
            "window_active": False,
            "vmax_active": None,
        }

    vmax = float(vmax_active)
    alpha = float(config.alpha)
    tol = float(config.tolerance)

    if vmax <= 0.0:
        raise ValueError(f"Velocity CBF requires vmax > 0, got {vmax}.")
    if alpha <= 0.0:
        raise ValueError(f"Velocity CBF requires alpha > 0, got {alpha}.")

    h = float(vmax * vmax - speed_sq)
    normal = 2.0 * v
    rhs = float(alpha * h)
    lhs = float(np.dot(normal, a_nom))
    violation = lhs - rhs

    if violation <= tol:
        return a_nom.copy(), {
            "h": h,
            "cbf_lhs": lhs,
            "cbf_rhs": rhs,
            "active": False,
            "correction_norm": 0.0,
            "speed": speed,
            "window_active": True,
            "vmax_active": vmax,
        }

    denom = float(np.dot(normal, normal))
    if denom <= tol:
        # At zero velocity the CBF constraint has no acceleration authority,
        # and h is strictly positive for vmax > 0.  Leave the command unchanged.
        return a_nom.copy(), {
            "h": h,
            "cbf_lhs": lhs,
            "cbf_rhs": rhs,
            "active": False,
            "correction_norm": 0.0,
            "speed": speed,
            "window_active": True,
            "vmax_active": vmax,
        }

    correction = (violation / denom) * normal
    a_safe = a_nom - correction
    lhs_safe = float(np.dot(normal, a_safe))

    return a_safe, {
        "h": h,
        "cbf_lhs": lhs_safe,
        "cbf_rhs": rhs,
        "active": True,
        "correction_norm": float(np.linalg.norm(correction)),
        "speed": speed,
        "window_active": True,
        "vmax_active": vmax,
    }


def filter_angular_velocity_acceleration(beta_nom, omega, config, t=None):
    """
    Project angular acceleration onto the angular-velocity CBF half-space.

    For h(omega) = omega_max^2 - ||omega||^2 and omega_dot = beta:

        h_dot + alpha h >= 0
        -2 omega^T beta + alpha h >= 0
        2 omega^T beta <= alpha h.
    """
    beta_nom = np.asarray(beta_nom, dtype=float).reshape(3)
    omega = np.asarray(omega, dtype=float).reshape(3)
    omega_norm_sq = float(np.dot(omega, omega))
    omega_norm = float(np.sqrt(omega_norm_sq))

    disabled_diag = {
        "h": np.inf,
        "cbf_lhs": -np.inf,
        "cbf_rhs": np.inf,
        "active": False,
        "correction_norm": 0.0,
        "omega_norm": omega_norm,
        "enabled": False,
    }

    if config is None or not config.enabled:
        return beta_nom.copy(), disabled_diag

    if t is not None and (t < config.t_start - 1e-12 or t > config.t_end + 1e-12):
        omega_max = float(config.omega_max)
        h = omega_max * omega_max - omega_norm_sq
        diag = disabled_diag.copy()
        diag["h"] = h
        return beta_nom.copy(), diag

    omega_max = float(config.omega_max)
    alpha = float(config.alpha)
    tol = float(config.tolerance)
    if omega_max <= 0.0:
        raise ValueError(
            f"Angular velocity CBF requires omega_max > 0, got {omega_max}."
        )
    if alpha <= 0.0:
        raise ValueError(f"Angular velocity CBF requires alpha > 0, got {alpha}.")

    h = float(omega_max * omega_max - omega_norm_sq)
    normal = 2.0 * omega
    rhs = float(alpha * h)
    lhs = float(np.dot(normal, beta_nom))
    violation = lhs - rhs

    if violation <= tol:
        return beta_nom.copy(), {
            "h": h,
            "cbf_lhs": lhs,
            "cbf_rhs": rhs,
            "active": False,
            "correction_norm": 0.0,
            "omega_norm": omega_norm,
            "enabled": True,
        }

    denom = float(np.dot(normal, normal))
    if denom <= tol:
        return beta_nom.copy(), {
            "h": h,
            "cbf_lhs": lhs,
            "cbf_rhs": rhs,
            "active": False,
            "correction_norm": 0.0,
            "omega_norm": omega_norm,
            "enabled": True,
        }

    correction = (violation / denom) * normal
    beta_safe = beta_nom - correction
    lhs_safe = float(np.dot(normal, beta_safe))

    return beta_safe, {
        "h": h,
        "cbf_lhs": lhs_safe,
        "cbf_rhs": rhs,
        "active": True,
        "correction_norm": float(np.linalg.norm(correction)),
        "omega_norm": omega_norm,
        "enabled": True,
    }


def _quat_normalize(q):
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_exp(v):
    v = np.asarray(v, dtype=float).reshape(3)
    half_angle = np.linalg.norm(v)
    if half_angle < 1e-10:
        return _quat_normalize(np.array([1.0, v[0], v[1], v[2]]))
    axis = v / half_angle
    return np.array([
        np.cos(half_angle),
        *(np.sin(half_angle) * axis),
    ])


def _quat_integrate_left(q, omega, dt):
    delta = 0.5 * np.asarray(omega, dtype=float).reshape(3) * float(dt)
    return _quat_normalize(_quat_mul(_quat_exp(delta), q))


def _left_quat_kinematic_matrix(q):
    """
    Return B(q) such that q_dot = B(q) omega for q_dot = 0.5 [0,omega] ⊗ q.
    """
    qw, qx, qy, qz = q
    return 0.5 * np.array([
        [-qx, -qy, -qz],
        [ qw,  qz, -qy],
        [-qz,  qw,  qx],
        [ qy, -qx,  qw],
    ])


def _orientation_barrier_terms(q, omega, config):
    q_ref = _quat_normalize(config.q_ref)
    q_eval = _quat_normalize(q)
    if float(np.dot(q_eval, q_ref)) < 0.0:
        q_eval = -q_eval

    max_angle = float(config.max_angle_rad)
    c = float(np.cos(0.5 * max_angle))
    r = float(np.dot(q_eval, q_ref))
    h = r * r - c * c

    B = _left_quat_kinematic_matrix(q_eval)
    qdot = B @ np.asarray(omega, dtype=float).reshape(3)
    hdot = 2.0 * r * float(np.dot(q_ref, qdot))
    control_row = 2.0 * r * (q_ref @ B)

    angle = 2.0 * np.arccos(np.clip(abs(r), -1.0, 1.0))
    return q_eval, h, hdot, control_row, angle


def orientation_hocbf_values(q, omega, config):
    """Return h, hdot, psi1, and cone angle for diagnostics."""
    _, h, hdot, _, angle = _orientation_barrier_terms(q, omega, config)
    psi1 = hdot + float(config.alpha1) * h
    return h, hdot, psi1, angle


def filter_orientation_acceleration(beta_nom, q, omega, config, t=None):
    """
    Project angular acceleration onto a quaternion cone HOCBF half-space.

    Barrier:
        h(q) = (q^T q_ref)^2 - cos(max_angle/2)^2

    HOCBF:
        psi1 = h_dot + alpha1 h
        psi1_dot + alpha2 psi1 >= 0

    The control coefficient Lg Lf h is analytic.  The drift part of h_ddot is
    estimated by a small zero-control finite difference under the same
    quaternion kinematics used by OrientationDMP.  The resulting inequality is
    affine in angular acceleration beta.
    """
    beta_nom = np.asarray(beta_nom, dtype=float).reshape(3)
    omega = np.asarray(omega, dtype=float).reshape(3)

    if config is None or not config.enabled:
        return beta_nom.copy(), {
            "h": np.inf,
            "hdot": 0.0,
            "psi1": np.inf,
            "hocbf_lhs": np.inf,
            "hocbf_rhs": -np.inf,
            "active": False,
            "correction_norm": 0.0,
            "angle_rad": 0.0,
            "enabled": False,
        }

    if t is not None and (t < config.t_start - 1e-12 or t > config.t_end + 1e-12):
        h, hdot, psi1, angle = orientation_hocbf_values(q, omega, config)
        return beta_nom.copy(), {
            "h": h,
            "hdot": hdot,
            "psi1": psi1,
            "hocbf_lhs": np.inf,
            "hocbf_rhs": -np.inf,
            "active": False,
            "correction_norm": 0.0,
            "angle_rad": angle,
            "enabled": False,
        }

    if config.max_angle_rad <= 0.0 or config.max_angle_rad > np.pi:
        raise ValueError(
            "Orientation HOCBF requires max_angle_rad in (0, pi], "
            f"got {config.max_angle_rad}."
        )
    if config.alpha1 <= 0.0 or config.alpha2 <= 0.0:
        raise ValueError("Orientation HOCBF alpha1 and alpha2 must be positive.")

    q_eval, h, hdot, control_row, angle = _orientation_barrier_terms(
        q, omega, config
    )
    psi1 = hdot + float(config.alpha1) * h

    eps = max(1e-8, float(config.finite_difference_dt))
    q_next = _quat_integrate_left(q_eval, omega, eps)
    _, hdot_next, _, _ = orientation_hocbf_values(q_next, omega, config)
    drift_hddot = (hdot_next - hdot) / eps

    drift = (
        drift_hddot
        + (float(config.alpha1) + float(config.alpha2)) * hdot
        + float(config.alpha1) * float(config.alpha2) * h
    )
    rhs = -float(drift)
    lhs = float(np.dot(control_row, beta_nom))
    violation = rhs - lhs
    tol = float(config.tolerance)

    if violation <= tol:
        return beta_nom.copy(), {
            "h": h,
            "hdot": hdot,
            "psi1": psi1,
            "hocbf_lhs": lhs,
            "hocbf_rhs": rhs,
            "active": False,
            "correction_norm": 0.0,
            "angle_rad": angle,
            "enabled": True,
        }

    denom = float(np.dot(control_row, control_row))
    if denom <= tol:
        return beta_nom.copy(), {
            "h": h,
            "hdot": hdot,
            "psi1": psi1,
            "hocbf_lhs": lhs,
            "hocbf_rhs": rhs,
            "active": False,
            "correction_norm": 0.0,
            "angle_rad": angle,
            "enabled": True,
        }

    correction = (violation / denom) * control_row
    beta_safe = beta_nom + correction
    lhs_safe = float(np.dot(control_row, beta_safe))

    return beta_safe, {
        "h": h,
        "hdot": hdot,
        "psi1": psi1,
        "hocbf_lhs": lhs_safe,
        "hocbf_rhs": rhs,
        "active": True,
        "correction_norm": float(np.linalg.norm(correction)),
        "angle_rad": angle,
        "enabled": True,
    }
