"""
Hard obstacle handling by post-rollout geometric trajectory deformation.

The DMP rollout is left untouched.  We only deform the sampled trajectory
after rollout and before robustness evaluation:

    traj_nominal = dmp.rollout(...)
    traj_safe    = deform(traj_nominal, obstacles)

The deformation is purely geometric:
  1) detect colliding trajectory samples
  2) radially project colliding samples to a safe boundary
  3) smooth only the correction vectors with localized, normalized
     multi-Gaussian averaging
  4) apply a tiny outward cleanup only if smoothing leaves residual
     penetration

This avoids repulsive-force dynamics and keeps obstacle handling modular.
"""

import numpy as np


class ObstacleProjector:
    """
    Deforms a position trajectory around a list of obstacles.

    Parameters
    ----------
    obstacles : list of dict, each with keys:
        "center" : array-like (3,)   — obstacle centre in world frame
        "radius" : float             — safe clearance radius
        "geometry" : str (optional)  — "sphere" (default) or
                                        "cylinder_infinite"
        "margin" : float (optional)  — collision margin
        "eps"    : float (optional)  — projection buffer
        "sigma"  : float (optional)  — Gaussian width (timesteps)
        "window_factor" : float (optional) — smoothing window = factor * sigma
    """

    def __init__(self, obstacles=None, margin=0.02, eps=0.01, sigma=12.0,
                 window_factor=4.0):
        self.obstacles = []
        self.margin = float(margin)
        self.eps = float(eps)
        self.sigma = float(sigma)
        self.window_factor = float(window_factor)
        self.last_debug = None
        if obstacles is not None:
            for obs in obstacles:
                self.add(
                    obs["center"],
                    obs["radius"],
                    obs.get("geometry", "sphere"),
                    margin=obs.get("margin", self.margin),
                    eps=obs.get("eps", self.eps),
                    sigma=obs.get("sigma", self.sigma),
                    window_factor=obs.get("window_factor", self.window_factor),
                )

    def add(self, center, radius, geometry="sphere", margin=0.02, eps=0.01,
            sigma=12.0, window_factor=4.0):
        self.obstacles.append({
            "center": np.asarray(center, dtype=float),
            "radius": float(radius),
            "geometry": str(geometry),
            "margin": float(margin),
            "eps": float(eps),
            "sigma": float(sigma),
            "window_factor": float(window_factor),
        })

    @staticmethod
    def _group_consecutive(indices):
        if len(indices) == 0:
            return []
        indices = np.asarray(indices, dtype=int)
        split_idx = np.where(np.diff(indices) > 1)[0] + 1
        return [chunk.astype(int) for chunk in np.split(indices, split_idx)]

    @staticmethod
    def _finite_difference_velocity(pos, dt):
        vel = np.empty_like(pos)
        T = pos.shape[0]
        if T >= 2:
            vel[0] = (pos[1] - pos[0]) / dt
            vel[-1] = (pos[-1] - pos[-2]) / dt
            if T > 2:
                vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
        else:
            vel[:] = 0.0
        return vel

    @staticmethod
    def _finite_difference_acceleration(vel, dt):
        acc = np.empty_like(vel)
        T = vel.shape[0]
        if T >= 2:
            acc[0] = (vel[1] - vel[0]) / dt
            acc[-1] = (vel[-1] - vel[-2]) / dt
            if T > 2:
                acc[1:-1] = (vel[2:] - vel[:-2]) / (2.0 * dt)
        else:
            acc[:] = 0.0
        return acc

    @staticmethod
    def _adaptive_sigma(base_sigma, seg_len):
        if base_sigma <= 0.0:
            return 0.0
        # Keep deformation local, but spread support a bit wider than the
        # first version so the boundary-enter / boundary-exit corners soften.
        return max(2.0, min(float(base_sigma), 0.75 * float(max(seg_len, 1)) + 1.0))

    @staticmethod
    def _safe_divide(delta_sum, weight_sum):
        out = np.zeros_like(delta_sum)
        valid = weight_sum > 1e-8
        out[valid] = delta_sum[valid] / weight_sum[valid, None]
        return out

    @staticmethod
    def _project_point(point, center, safe_radius, geometry, default_dir):
        if geometry == "sphere":
            diff = point - center
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                direction = diff / dist
            else:
                direction = default_dir
            proj = center + safe_radius * direction
            return proj, direction, dist

        if geometry == "cylinder_infinite":
            diff_xy = point[:2] - center[:2]
            dist_xy = np.linalg.norm(diff_xy)
            if dist_xy > 1e-9:
                direction_xy = diff_xy / dist_xy
            else:
                direction_xy = default_dir[:2]
            proj = point.copy()
            proj[:2] = center[:2] + safe_radius * direction_xy
            direction = np.array([direction_xy[0], direction_xy[1], 0.0])
            return proj, direction, dist_xy

        raise ValueError(f"Unsupported obstacle geometry: {geometry}")

    def project(self, pos, vel, dt, return_debug=False):
        """
        Project positions outside all obstacles and recompute velocity.

        Parameters
        ----------
        pos : (T, 3) ndarray  — DMP-generated positions (may violate constraints)
        vel : (T, 3) ndarray  — DMP-generated velocities
        dt  : float           — timestep (used for velocity recomputation)

        Returns
        -------
        pos_safe : (T, 3)  — deformed positions after projection + smoothing
        vel_safe : (T, 3)  — finite-difference velocity on deformed path
        debug    : dict    — optional deformation diagnostics
        """
        if not self.obstacles:
            debug = {
                "applied": False,
                "nominal_position": pos.copy(),
                "deformed_position": pos.copy(),
                "deformed_velocity": vel.copy(),
                "deformed_acceleration": self._finite_difference_acceleration(vel, dt),
                "delta_smooth": np.zeros_like(pos),
                "cleanup_delta": np.zeros_like(pos),
                "weight_sum": np.zeros(pos.shape[0], dtype=float),
                "obstacles": [],
                "collision_indices": np.array([], dtype=int),
            }
            self.last_debug = debug
            if return_debug:
                return pos.copy(), vel.copy(), debug
            return pos.copy(), vel.copy()

        pos_nominal = pos.copy()
        pos_safe = pos.copy()

        default_dir = np.array([0.70710678, 0.70710678, 0.0])
        T = pos_safe.shape[0]
        idxs = np.arange(T)
        delta_sum = np.zeros_like(pos_safe)
        weight_sum = np.zeros(T, dtype=float)
        all_collision_indices = []
        obstacle_debug = []
        projected_aligned = np.full_like(pos_nominal, np.nan)
        collision_mask = np.zeros(T, dtype=bool)

        for obs in self.obstacles:
            c = obs["center"]
            r = obs["radius"]
            geometry = obs.get("geometry", "sphere")
            margin = float(obs.get("margin", self.margin))
            eps = float(obs.get("eps", self.eps))
            sigma_base = float(obs.get("sigma", self.sigma))
            window_factor = float(obs.get("window_factor", self.window_factor))
            safe_radius = r + eps

            obs_debug = {
                "center": c.copy(),
                "radius": float(r),
                "margin": margin,
                "safe_radius": safe_radius,
                "geometry": geometry,
                "collision_indices": [],
                "segments": [],
                "projected_points": [],
                "correction_vectors": [],
            }

            if geometry == "sphere":
                diff = pos_nominal - c
                d = np.linalg.norm(diff, axis=1)
                inside = d < (r + margin)
            elif geometry == "cylinder_infinite":
                diff_xy = pos_nominal[:, :2] - c[:2]
                d_xy = np.linalg.norm(diff_xy, axis=1)
                inside = d_xy < (r + margin)
            else:
                raise ValueError(f"Unsupported obstacle geometry: {geometry}")

            hit_idx = np.flatnonzero(inside)
            obs_debug["collision_indices"] = hit_idx.astype(int).tolist()
            if hit_idx.size == 0:
                obstacle_debug.append(obs_debug)
                continue

            all_collision_indices.extend(hit_idx.tolist())
            collision_mask[hit_idx] = True
            for seg in self._group_consecutive(hit_idx):
                sigma_seg = self._adaptive_sigma(sigma_base, len(seg))
                window = int(np.ceil(window_factor * sigma_seg)) if window_factor > 0.0 else 0
                seg_debug = {
                    "indices": seg.astype(int).tolist(),
                    "sigma": float(sigma_seg),
                    "window": int(window),
                }
                obs_debug["segments"].append(seg_debug)

                for t_idx in seg:
                    proj, _, _ = self._project_point(
                        pos_nominal[t_idx], c, safe_radius, geometry, default_dir
                    )
                    delta = proj - pos_nominal[t_idx]

                    k0 = max(0, t_idx - window)
                    k1 = min(T, t_idx + window + 1)
                    kk = idxs[k0:k1]
                    if sigma_seg <= 1e-8 or window == 0:
                        w = np.ones(1, dtype=float)
                        kk = np.array([t_idx], dtype=int)
                        k0 = t_idx
                        k1 = t_idx + 1
                    else:
                        w = np.exp(-0.5 * ((kk - t_idx) / sigma_seg) ** 2)

                    delta_sum[kk] += w[:, None] * delta[None, :]
                    weight_sum[kk] += w

                    projected_aligned[t_idx] = proj
                    obs_debug["projected_points"].append(proj.copy())
                    obs_debug["correction_vectors"].append(delta.copy())

            obstacle_debug.append(obs_debug)

        delta_smooth = self._safe_divide(delta_sum, weight_sum)
        # Keep endpoints fixed unless they are themselves colliding and later
        # need the tiny cleanup pass for safety.
        if T >= 1:
            delta_smooth[0] = 0.0
            delta_smooth[-1] = 0.0

        if np.any(weight_sum > 0.0):
            pos_safe = pos_nominal + delta_smooth

        cleanup_delta = np.zeros_like(pos_safe)
        for _ in range(2):
            cleanup_applied = False
            for obs in self.obstacles:
                c = obs["center"]
                r = obs["radius"]
                geometry = obs.get("geometry", "sphere")
                eps = float(obs.get("eps", self.eps))
                safe_radius = r + eps

                if geometry == "sphere":
                    dist = np.linalg.norm(pos_safe - c, axis=1)
                    residual = dist < safe_radius
                elif geometry == "cylinder_infinite":
                    dist = np.linalg.norm(pos_safe[:, :2] - c[:2], axis=1)
                    residual = dist < safe_radius
                else:
                    raise ValueError(f"Unsupported obstacle geometry: {geometry}")

                if not np.any(residual):
                    continue

                for t_idx in np.flatnonzero(residual):
                    proj, _, _ = self._project_point(
                        pos_safe[t_idx], c, safe_radius, geometry, default_dir
                    )
                    tiny_delta = proj - pos_safe[t_idx]
                    if np.linalg.norm(tiny_delta) <= 1e-12:
                        continue
                    pos_safe[t_idx] += tiny_delta
                    cleanup_delta[t_idx] += tiny_delta
                    cleanup_applied = True

            if not cleanup_applied:
                break

        vel_safe = self._finite_difference_velocity(pos_safe, dt)
        acc_safe = self._finite_difference_acceleration(vel_safe, dt)

        collision_indices = np.array(sorted(set(all_collision_indices)), dtype=int)
        debug = {
            "applied": bool(collision_indices.size > 0),
            "nominal_position": pos_nominal,
            "deformed_position": pos_safe.copy(),
            "deformed_velocity": vel_safe.copy(),
            "deformed_acceleration": acc_safe,
            "delta_smooth": delta_smooth,
            "cleanup_delta": cleanup_delta,
            "weight_sum": weight_sum,
            "support_mask": weight_sum > 1e-8,
            "collision_mask": collision_mask,
            "projected_position": projected_aligned,
            "obstacles": obstacle_debug,
            "collision_indices": collision_indices,
        }
        self.last_debug = debug

        if return_debug:
            return pos_safe, vel_safe, debug
        return pos_safe, vel_safe
