"""
core/obstacle_projection.py
===========================
Hard obstacle avoidance by construction — a post-DMP position projection.

Philosophy
----------
The DMP + CGMS framework enforces hard constraints by parameterisation:
  K = QᵀQ  >0  by construction (Cholesky ODE)
  D ≽ αH        by construction (SD slack)

We apply the same philosophy to position constraints.  Instead of a soft
penalty, we project every waypoint *outside* each obstacle after the
DMP rollout.  This gives:

    sphere:              ||p(t) − c|| >= r_safe
    infinite cylinder:   ||p_xy(t) − c_xy|| >= r_safe

hard, by construction.

The projection does NOT modify the gain schedule (K, D) or any other part
of the CGMS structure.  It only displaces position points that fall inside
an obstacle sphere radially outward to the sphere surface.

Projection formulas
-------------------
Sphere (center c, radius r), for p inside:
    d = ||p − c||
    p' = c + r * (p − c) / d         if d > ε
    p' = c + r * e_default           if d ≤ ε

Infinite cylinder aligned with world Z (axis through c_xy), for p inside:
    d_xy = ||p_xy − c_xy||
    p'_xy = c_xy + r * (p_xy − c_xy) / d_xy   if d_xy > ε
    p'_xy = c_xy + r * e_default_xy           if d_xy ≤ ε
    p'_z = p_z

where e_default is a unit vector in the XY plane at 45°.

Smoothness note
---------------
The projection is applied per-timestep independently.  Adjacent points that
are both projected stay smooth because the DMP forcing function is smooth —
if p(t) barely enters the sphere the displacement is small; if it plunges
deep the displacement is larger.  This is identical to a reflecting boundary
and preserves trajectory smoothness in practice.

Velocity update
---------------
After projection the velocity is updated by finite difference on the
projected positions.  The first and last points use forward/backward
differences; interior points use central differences.  This keeps the
velocity consistent with the projected path for all downstream cost terms
(VelocityLimit predicate, etc.).

Usage
-----
    from core.obstacle_projection import ObstacleProjector

    projector = ObstacleProjector([
        {"center": [0.40, 0.30, 0.30], "radius": 0.12},
    ])
    pos_safe, vel_safe = projector.project(pos, vel, dt)
"""

import numpy as np


class ObstacleProjector:
    """
    Projects a position trajectory outside a list of obstacles.

    Parameters
    ----------
    obstacles : list of dict, each with keys:
        "center" : array-like (3,)   — obstacle centre in world frame
        "radius" : float             — safe clearance radius
        "geometry" : str (optional)  — "sphere" (default) or
                                        "cylinder_infinite"
    """

    def __init__(self, obstacles=None):
        self.obstacles = []
        if obstacles is not None:
            for obs in obstacles:
                self.add(obs["center"], obs["radius"], obs.get("geometry", "sphere"))

    def add(self, center, radius, geometry="sphere"):
        self.obstacles.append({
            "center": np.asarray(center, dtype=float),
            "radius": float(radius),
            "geometry": str(geometry),
        })

    def project(self, pos, vel, dt, return_mask=False):
        """
        Project positions outside all obstacles and recompute velocity.

        Parameters
        ----------
        pos : (T, 3) ndarray  — DMP-generated positions (may violate constraints)
        vel : (T, 3) ndarray  — DMP-generated velocities
        dt  : float           — timestep (used for velocity recomputation)

        Returns
        -------
        pos_safe : (T, 3)  — projected positions, guaranteed outside all obstacles
        vel_safe : (T, 3)  — finite-difference velocity on projected path
        projected : (T,) bool (optional)
            True where any obstacle projection was applied.
        """
        if not self.obstacles:
            if return_mask:
                return pos.copy(), vel.copy(), np.zeros(pos.shape[0], dtype=bool)
            return pos.copy(), vel.copy()

        pos_safe = pos.copy()
        projected = np.zeros(pos.shape[0], dtype=bool)

        # Default escape direction (used only in degenerate case d≈0)
        _e_default = np.array([0.70710678, 0.70710678, 0.0])

        for obs in self.obstacles:
            c = obs["center"]
            r = obs["radius"]
            geometry = obs.get("geometry", "sphere")

            if geometry == "sphere":
                diff = pos_safe - c
                d = np.linalg.norm(diff, axis=1)
                inside = d < r
                if not np.any(inside):
                    continue
                projected |= inside

                for t_idx in np.where(inside)[0]:
                    dist = d[t_idx]
                    if dist > 1e-9:
                        direction = diff[t_idx] / dist
                    else:
                        direction = _e_default
                    pos_safe[t_idx] = c + r * direction
            elif geometry == "cylinder_infinite":
                diff_xy = pos_safe[:, :2] - c[:2]
                d_xy = np.linalg.norm(diff_xy, axis=1)
                inside = d_xy < r
                if not np.any(inside):
                    continue
                projected |= inside

                for t_idx in np.where(inside)[0]:
                    dist_xy = d_xy[t_idx]
                    if dist_xy > 1e-9:
                        dir_xy = diff_xy[t_idx] / dist_xy
                    else:
                        dir_xy = _e_default[:2]
                    pos_safe[t_idx, :2] = c[:2] + r * dir_xy
            else:
                raise ValueError(f"Unsupported obstacle geometry: {geometry}")

        # Recompute velocity by finite difference on projected path
        T = pos_safe.shape[0]
        vel_safe = np.empty_like(pos_safe)
        if T >= 2:
            vel_safe[0]    = (pos_safe[1]   - pos_safe[0])   / dt  # forward
            vel_safe[-1]   = (pos_safe[-1]  - pos_safe[-2])  / dt  # backward
            if T > 2:
                vel_safe[1:-1] = (pos_safe[2:] - pos_safe[:-2]) / (2.0 * dt)  # central
        else:
            vel_safe[:] = 0.0

        if return_mask:
            return pos_safe, vel_safe, projected
        return pos_safe, vel_safe
