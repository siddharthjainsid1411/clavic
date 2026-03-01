"""
Multi-phase certified policy.

Chains N DMP segments end-to-end.  Each segment has its own start/end,
duration, and learnable weights.  The traces are concatenated into a
single Trace so the existing temporal-logic compiler works unchanged.

Phase layout (example — 3 phases):
    Phase 1: start → waypoint       (duration_1)
    Phase 2: waypoint → waypoint    (hold — duration_2, same start/end)
    Phase 3: waypoint → goal        (duration_3)

Total horizon = sum(duration_i).

Theta layout:
    [phase_1 weights | phase_2 weights | phase_3 weights]
Each phase has its own theta_dim (may differ if n_bfs differ, but
we keep them identical for simplicity).
"""

import numpy as np
from core.cgms.dmp_with_gain import DMPWithGainScheduling
from core.cgms.dynamical_systems import DynamicalSystems
from core.certified_policy import Trace


class MultiPhaseCertifiedPolicy:
    """
    Chains multiple DMP segments.  Each phase is an independent
    DMPWithGainScheduling instance with its own weights.
    """

    def __init__(self, phases):
        """
        Parameters
        ----------
        phases : list of dict
            Each dict contains:
                start       : np.ndarray (3,)
                end         : np.ndarray (3,)
                duration    : float        (seconds)
                n_bfs_traj  : int          (default 51)
                n_bfs_slack : int          (default 7)
        """
        self.phases = phases
        self.dmps = []
        self.sizes_list = []     # (n_traj, n_sd, n_sk) per phase
        self.theta_dims = []     # theta_dim per phase
        self.offsets = []        # cumulative offsets into flat theta

        DT    = 0.01
        ALPHA = 0.05
        K0    = 200.0
        D0    = 30.0
        H     = np.eye(3)

        offset = 0
        for p in phases:
            dmp = DMPWithGainScheduling(
                start=np.asarray(p["start"], float),
                end=np.asarray(p["end"], float),
                tau=p["duration"],
                dt=DT,
                n_bfs_traj=p.get("n_bfs_traj", 51),
                n_bfs_slack=p.get("n_bfs_slack", 7),
                K0=K0, D0=D0, alpha=ALPHA, H=H,
            )
            theta_init, n_traj, n_sd, n_sk = dmp.initial_weights()
            dim = len(theta_init)

            self.dmps.append(dmp)
            self.sizes_list.append((n_traj, n_sd, n_sk))
            self.theta_dims.append(dim)
            self.offsets.append(offset)
            offset += dim

        self.total_theta_dim = offset
        self.DT = DT

    # ------------------------------------------------------------------ #
    def parameter_dimension(self):
        return self.total_theta_dim

    # ------------------------------------------------------------------ #
    def structured_sigma(self, sigma_traj_xy=5.0, sigma_traj_z=5.0,
                         sigma_sd=5.0, sigma_sk=5.0):
        """Per-parameter exploration noise — concatenated over all phases."""
        parts = []
        for idx, (dmp, sizes) in enumerate(zip(self.dmps, self.sizes_list)):
            n_traj, n_sd, n_sk = sizes
            n_per_axis = n_traj // 3
            sigma = np.empty(self.theta_dims[idx])
            off = 0
            sigma[off:off + n_per_axis] = sigma_traj_xy; off += n_per_axis
            sigma[off:off + n_per_axis] = sigma_traj_xy; off += n_per_axis
            sigma[off:off + n_per_axis] = sigma_traj_z;  off += n_per_axis
            sigma[off:off + n_sd]       = sigma_sd;       off += n_sd
            sigma[off:off + n_sk]       = sigma_sk;       off += n_sk
            parts.append(sigma)
        return np.concatenate(parts)

    # ------------------------------------------------------------------ #
    def rollout(self, theta):
        """
        Run all DMP phases in sequence, stitching position & velocity
        continuously.  Returns a single Trace covering [0, total_horizon].
        """
        all_time = []
        all_pos  = []
        all_vel  = []
        all_K    = []
        all_D    = []
        all_raw_sk = []
        all_raw_sd = []

        t_offset = 0.0    # global time offset for concatenation

        for idx, (dmp, sizes, tdim) in enumerate(
            zip(self.dmps, self.sizes_list, self.theta_dims)
        ):
            off = self.offsets[idx]
            theta_phase = theta[off:off + tdim]

            # Ensure DMP internal timing is correct
            dur = self.phases[idx]["duration"]
            dmp.tau = dur
            dmp.ts  = np.arange(0.0, dur + 1e-12, dmp.dt)
            dmp.T   = dmp.ts.size
            dmp.ds  = DynamicalSystems(dur)

            # Extract raw weights before clipping
            n_traj, n_sd, n_sk = sizes
            raw_sd = theta_phase[n_traj:n_traj + n_sd].copy()
            raw_sk = theta_phase[n_traj + n_sd:n_traj + n_sd + n_sk].copy()

            dmp.set_theta(theta_phase, sizes)
            plan = dmp.rollout_traj()

            ts    = plan["ts"] + t_offset
            y     = plan["y_des"]
            yd    = plan["yd_des"]
            K     = plan["K"]
            D     = plan["D"]

            # For phases after the first, drop the first timestep to
            # avoid duplicate time=boundary points.
            if idx > 0:
                ts = ts[1:]
                y  = y[1:]
                yd = yd[1:]
                K  = K[1:]
                D  = D[1:]

            all_time.append(ts)
            all_pos.append(y)
            all_vel.append(yd)
            all_K.append(K)
            all_D.append(D)
            all_raw_sk.append(raw_sk)
            all_raw_sd.append(raw_sd)

            t_offset = ts[-1]

        trace = Trace(
            time=np.concatenate(all_time),
            position=np.concatenate(all_pos),
            velocity=np.concatenate(all_vel),
            gains={
                "K": np.concatenate(all_K),
                "D": np.concatenate(all_D),
            },
            raw_sk_weights=np.concatenate(all_raw_sk),
            raw_sd_weights=np.concatenate(all_raw_sd),
        )
        return trace
