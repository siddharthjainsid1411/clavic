"""
Orientation DMP — spring-damper system in quaternion tangent space.

Operates on the log-map error between current and goal quaternion.
Uses same RBF basis as position DMP for the forcing term.

Equations (in tangent space ℝ³):
    e(t) = log(q_goal(t)⁻¹ ⊗ q(t))         — orientation error
    τ² ω̇ = k_ori · e(t) - τ · d_ori · ω + γ(t) · f_ori(s)
    q̇   integrated via exponential map

Convention:  q = [w, x, y, z]  (scalar-first)
"""

import numpy as np
from core.cbf_filter import (
    filter_angular_velocity_acceleration,
    filter_orientation_acceleration,
)
from .quat_utils import (
    quat_normalize, quat_error, quat_integrate, quat_slerp,
)
from .function_approximator import FunctionApproximatorRBFN
from .dynamical_systems import DynamicalSystems


class OrientationDMP:
    """
    Orientation DMP with RBF forcing in quaternion log-space.
    """

    def __init__(self, q_start, q_end, tau, dt, n_bfs_ori=15):
        """
        Parameters
        ----------
        q_start   : array (4,)  — start quaternion [w,x,y,z]
        q_end     : array (4,)  — goal quaternion  [w,x,y,z]
        tau       : float       — phase duration (seconds)
        dt        : float       — timestep
        n_bfs_ori : int         — number of RBF basis functions for orientation forcing
        """
        self.q_start = quat_normalize(np.asarray(q_start, float))
        self.q_end   = quat_normalize(np.asarray(q_end, float))
        self.tau     = float(tau)
        self.dt      = float(dt)
        self.ts      = np.arange(0.0, self.tau + 1e-12, self.dt)
        self.T       = self.ts.size
        self.n_bfs   = n_bfs_ori

        # Spring-damper gains (critically damped in log-space)
        self.d_ori = 20.0
        self.k_ori = (self.d_ori ** 2) / 4.0

        self.ds = DynamicalSystems(self.tau)
        self.time_offset = 0.0
        self.orientation_hocbf_configs = []
        self.angular_velocity_cbf_configs = []
        self.omega_init = None

        # 3 independent RBF networks — one per tangent-space axis
        self.rbf_ori = [
            FunctionApproximatorRBFN(n_bfs_ori, normalize=True, intersection_height=0.95)
            for _ in range(3)
        ]

        # Train RBFs to reproduce zero forcing (identity rollout = SLERP-like)
        phase = self.ds.time_system(self.ts)
        zero_forcing = np.zeros((self.T, 3))
        for i in range(3):
            self.rbf_ori[i].train(phase, zero_forcing[:, i])

    # ------------------------------------------------------------------ #
    def n_weights(self):
        """Total learnable weights: n_bfs × 3 axes."""
        return self.n_bfs * 3

    def initial_weights(self):
        """Return concatenated weight vector [axis_x | axis_y | axis_z]."""
        return np.concatenate([r.W.ravel() for r in self.rbf_ori])

    def set_weights(self, w_ori):
        """Set orientation RBF weights from flat vector."""
        off = 0
        for r in self.rbf_ori:
            n = r.W.size
            r.W = w_ori[off:off + n].reshape(r.W.shape)
            off += n

    # ------------------------------------------------------------------ #
    def _goal_at_time(self, t):
        """
        Time-varying goal quaternion.

        Default: constant q_end for the whole phase.
        Override this for time-varying goals (e.g., SLERP transition
        in the last N seconds).
        """
        return self.q_end.copy()

    # ------------------------------------------------------------------ #
    def rollout(self):
        """
        Integrate orientation DMP using RK4 in log-space.

        Returns
        -------
        dict with keys:
            q_des   : ndarray (T, 4)  — desired quaternion trajectory
            omega   : ndarray (T, 3)  — angular velocity trajectory
        """
        ts = self.ts
        T  = self.T
        tau = self.tau

        q_traj = np.zeros((T, 4))
        omega  = np.zeros((T, 3))
        q_traj[0] = self.q_start.copy()
        if self.omega_init is not None:
            omega[0] = np.asarray(self.omega_init, dtype=float).reshape(3)
        hocbf_h = np.full(T, np.nan)
        hocbf_hdot = np.full(T, np.nan)
        hocbf_psi1 = np.full(T, np.nan)
        hocbf_lhs = np.full(T, np.nan)
        hocbf_rhs = np.full(T, np.nan)
        hocbf_active = np.zeros(T, dtype=bool)
        hocbf_enabled = np.zeros(T, dtype=bool)
        hocbf_correction = np.zeros(T)
        hocbf_angle = np.full(T, np.nan)
        av_h = np.full(T, np.nan)
        av_lhs = np.full(T, np.nan)
        av_rhs = np.full(T, np.nan)
        av_active = np.zeros(T, dtype=bool)
        av_enabled = np.zeros(T, dtype=bool)
        av_correction = np.zeros(T)
        av_norm = np.zeros(T)
        av_projection_active = np.zeros(T, dtype=bool)
        av_projection_correction = np.zeros(T)

        def _omega_limit_at(global_t):
            limits = []
            for cfg in self.angular_velocity_cbf_configs:
                if not cfg.enabled:
                    continue
                if global_t < cfg.t_start - 1e-12 or global_t > cfg.t_end + 1e-12:
                    continue
                limits.append(float(cfg.omega_max))
            return min(limits) if limits else None

        def _project_omega_if_needed(w, global_t):
            limit = _omega_limit_at(global_t)
            if limit is None:
                return w, False, 0.0
            norm = float(np.linalg.norm(w))
            if norm <= limit + 1e-12:
                return w, False, 0.0
            w_safe = w * (limit / (norm + 1e-12))
            return w_safe, True, float(np.linalg.norm(w - w_safe))

        def _dmp_accel(t, q, w):
            """Compute angular acceleration  ω̇  at state (q, ω)."""
            phase = self.ds.time_system(np.array([t]))[0]
            gate  = phase  # gating decays from 1→0

            # Orientation error in tangent space
            # e = log(q_goal⁻¹ ⊗ q) points FROM goal TO current
            # Spring should pull TOWARD goal → use -e
            q_goal = self._goal_at_time(t)
            e = quat_error(q, q_goal)  # ℝ³ log-error

            # RBF forcing (per axis)
            f_ori = np.array([
                self.rbf_ori[i].predict(phase)[0, 0] for i in range(3)
            ])

            # Spring-damper + forcing
            # τ² ω̇ = -k·e - τ·d·ω + gate·f_ori
            acc = (-self.k_ori * e - tau * self.d_ori * w + gate * f_ori) / (tau ** 2)
            global_t = self.time_offset + t
            active_diag = None
            for cfg in self.orientation_hocbf_configs:
                acc, diag = filter_orientation_acceleration(
                    acc, q, w, cfg, t=global_t
                )
                if diag.get("enabled", False):
                    active_diag = diag
            if active_diag is None:
                # Keep a useful diagnostic from the first configured cone even
                # outside its time window; if no config exists, mark disabled.
                if self.orientation_hocbf_configs:
                    _, active_diag = filter_orientation_acceleration(
                        acc, q, w, self.orientation_hocbf_configs[0], t=global_t
                    )
                else:
                    active_diag = {
                        "h": np.inf, "hdot": 0.0, "psi1": np.inf,
                        "hocbf_lhs": np.inf, "hocbf_rhs": -np.inf,
                        "active": False, "correction_norm": 0.0,
                        "angle_rad": 0.0, "enabled": False,
                    }
            angular_diag = None
            for cfg in self.angular_velocity_cbf_configs:
                acc, diag = filter_angular_velocity_acceleration(
                    acc, w, cfg, t=global_t
                )
                if diag.get("enabled", False):
                    angular_diag = diag
            if angular_diag is None:
                if self.angular_velocity_cbf_configs:
                    _, angular_diag = filter_angular_velocity_acceleration(
                        acc, w, self.angular_velocity_cbf_configs[0], t=global_t
                    )
                else:
                    angular_diag = {
                        "h": np.inf, "cbf_lhs": -np.inf, "cbf_rhs": np.inf,
                        "active": False, "correction_norm": 0.0,
                        "omega_norm": float(np.linalg.norm(w)), "enabled": False,
                    }
            return acc, active_diag, angular_diag

        for k in range(T - 1):
            t0 = ts[k]
            h  = ts[k + 1] - ts[k]
            q0 = q_traj[k]
            w0 = omega[k]

            # RK4 for angular velocity (ω)
            k1w, diag, av_diag = _dmp_accel(t0,           q0,                              w0)
            k1q_mid = quat_integrate(q0, w0 + 0.5 * h * k1w, 0.0)  # just use q0 for mid
            k2w, _, _ = _dmp_accel(t0 + 0.5 * h, q0,                              w0 + 0.5 * h * k1w)
            k3w, _, _ = _dmp_accel(t0 + 0.5 * h, q0,                              w0 + 0.5 * h * k2w)
            k4w, _, _ = _dmp_accel(t0 + h,        q0,                              w0 + h * k3w)

            omega_next = w0 + (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
            omega_next, proj_active, proj_corr = _project_omega_if_needed(
                omega_next, self.time_offset + ts[k + 1]
            )
            omega[k + 1] = omega_next

            # Integrate quaternion using midpoint angular velocity
            w_mid = 0.5 * (omega[k] + omega[k + 1])
            q_traj[k + 1] = quat_integrate(q0, w_mid, h)

            hocbf_h[k] = diag["h"]
            hocbf_hdot[k] = diag["hdot"]
            hocbf_psi1[k] = diag["psi1"]
            hocbf_lhs[k] = diag["hocbf_lhs"]
            hocbf_rhs[k] = diag["hocbf_rhs"]
            hocbf_active[k] = diag["active"]
            hocbf_enabled[k] = diag.get("enabled", False)
            hocbf_correction[k] = diag["correction_norm"]
            hocbf_angle[k] = diag["angle_rad"]
            av_h[k] = av_diag["h"]
            av_lhs[k] = av_diag["cbf_lhs"]
            av_rhs[k] = av_diag["cbf_rhs"]
            av_active[k] = av_diag["active"]
            av_enabled[k] = av_diag.get("enabled", False)
            av_correction[k] = av_diag["correction_norm"]
            av_norm[k] = av_diag["omega_norm"]
            av_projection_active[k + 1] = proj_active
            av_projection_correction[k + 1] = proj_corr

        _, diag, av_diag = _dmp_accel(ts[-1], q_traj[-1], omega[-1])
        hocbf_h[-1] = diag["h"]
        hocbf_hdot[-1] = diag["hdot"]
        hocbf_psi1[-1] = diag["psi1"]
        hocbf_lhs[-1] = diag["hocbf_lhs"]
        hocbf_rhs[-1] = diag["hocbf_rhs"]
        hocbf_active[-1] = diag["active"]
        hocbf_enabled[-1] = diag.get("enabled", False)
        hocbf_correction[-1] = diag["correction_norm"]
        hocbf_angle[-1] = diag["angle_rad"]
        av_h[-1] = av_diag["h"]
        av_lhs[-1] = av_diag["cbf_lhs"]
        av_rhs[-1] = av_diag["cbf_rhs"]
        av_active[-1] = av_diag["active"]
        av_enabled[-1] = av_diag.get("enabled", False)
        av_correction[-1] = av_diag["correction_norm"]
        av_norm[-1] = av_diag["omega_norm"]

        return {
            "q_des": q_traj,
            "omega": omega,
            "safety": {
                "orientation_hocbf": {
                    "enabled": hocbf_enabled,
                    "h": hocbf_h,
                    "hdot": hocbf_hdot,
                    "psi1": hocbf_psi1,
                    "lhs": hocbf_lhs,
                    "rhs": hocbf_rhs,
                    "active": hocbf_active,
                    "correction_norm": hocbf_correction,
                    "angle_rad": hocbf_angle,
                }
                ,
                "angular_velocity_cbf": {
                    "enabled": av_enabled,
                    "h": av_h,
                    "lhs": av_lhs,
                    "rhs": av_rhs,
                    "active": av_active,
                    "correction_norm": av_correction,
                    "omega_norm": av_norm,
                    "projection_active": av_projection_active,
                    "projection_correction_norm": av_projection_correction,
                },
            },
        }
