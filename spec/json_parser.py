import json
import numpy as np
from spec.taskspec import TaskSpec, Clause

# Predicates that represent spherical obstacles — used to auto-extract
# obstacle specs when modality="HARD".
_OBSTACLE_PREDICATES = {
    "ObstacleAvoidance":  ("obstacle_position", "safe_radius"),
    "HumanBodyExclusion": ("human_position",    "body_radius"),
}

_AVOIDANCE_GEOMETRY = {"sphere", "cylinder_infinite"}

def _geometry_from_modality(modality):
    """
    Deterministic geometry policy:
      HARD                  -> cylinder_infinite
      SOFT / PREFER / legacy REQUIRE -> sphere
    """
    return "cylinder_infinite" if str(modality).upper() == "HARD" else "sphere"


def _resolve_avoidance_geometry(raw_geometry, modality):
    """Return canonical geometry string, falling back to modality default."""
    if raw_geometry is None:
        return _geometry_from_modality(modality)
    geom = str(raw_geometry).strip().lower()
    if geom not in _AVOIDANCE_GEOMETRY:
        raise ValueError(
            f"avoidance_geometry must be one of {sorted(_AVOIDANCE_GEOMETRY)}, got {raw_geometry!r}."
        )
    return geom


def _normalize_modality(modality):
    """Map legacy modality names onto the HARD/SOFT architecture."""
    modality = str(modality).upper()
    if modality == "REQUIRE":
        return "SOFT"
    if modality == "PREFER":
        return "SOFT"
    return modality


def _compute_cover_from_shape_points(shape_points, geometry):
    """
    Build a conservative cover from raw object points.

    Parameters
    ----------
    shape_points : array-like, shape (N, 3)
    geometry     : "sphere" or "cylinder_infinite"

    Returns
    -------
    center : np.ndarray, shape (3,)
    radius : float
    """
    pts = np.asarray(shape_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 1:
        raise ValueError(
            "shape_points must be an array-like of shape (N, 3) with N >= 1"
        )

    center = np.mean(pts, axis=0)

    if geometry == "sphere":
        radial = np.linalg.norm(pts - center[None, :], axis=1)
    elif geometry == "cylinder_infinite":
        radial = np.linalg.norm(pts[:, :2] - center[None, :2], axis=1)
    else:
        raise ValueError(f"Unsupported geometry for shape cover: {geometry}")

    radius = float(np.max(radial))
    return center, radius

def load_taskspec_from_json(path):

    with open(path, "r") as f:
        data = json.load(f)

    horizon_sec = data["horizon_sec"]
    bindings = data.get("bindings", {})
    phases = data.get("phases", None)

    clauses = []
    hard_obstacle_specs = []

    for c in data["clauses"]:

        operator = c["type"]
        weight   = c["weight"]
        raw_modality = c["modality"]
        modality = _normalize_modality(raw_modality)  # "HARD" or "SOFT"

        time_window = None
        if "time_window" in c:
            time_window = tuple(c["time_window"])

        if operator in ["always", "eventually", "always_during", "eventually_during"]:

            predicate = c["predicate"]
            if predicate in ("VelocityLimit", "OrientationLimit", "AngularVelocityLimit") and str(raw_modality).upper() not in ("SOFT", "PREFER"):
                # Runtime safety predicates by default.  Legacy REQUIRE specs
                # are promoted to HARD CBF/HOCBF enforcement.
                modality = "HARD"
            parameters = extract_parameters(predicate, bindings)

            # For obstacle-like predicates, geometry is deterministic by modality.
            if predicate in _OBSTACLE_PREDICATES:
                center_key, radius_key = _OBSTACLE_PREDICATES[predicate]
                geometry = _resolve_avoidance_geometry(c.get("avoidance_geometry"), modality)
                parameters["geometry"] = geometry

                # Optional generic shape-driven radius extraction:
                #   <Predicate>.shape_points : [[x,y,z], ...]
                #   <Predicate>.shape_margin : nonnegative float (optional)
                # If provided, these override direct center/radius bindings.
                if "shape_points" in parameters:
                    center, radius = _compute_cover_from_shape_points(
                        parameters["shape_points"], geometry
                    )
                    margin = float(parameters.get("shape_margin", 0.0))
                    if margin < 0.0:
                        raise ValueError("shape_margin must be nonnegative")
                    radius = float(radius + margin)
                    parameters[center_key] = center
                    parameters[radius_key] = radius

            # ── HARD: extract obstacle spec for Layers 1+2 ──────────────
            hard_obstacle = None
            if modality == "HARD" and predicate in _OBSTACLE_PREDICATES:
                center_key, radius_key = _OBSTACLE_PREDICATES[predicate]
                center = parameters.get(center_key)
                radius = parameters.get(radius_key)
                geometry = _resolve_avoidance_geometry(c.get("avoidance_geometry"), modality)
                if center is not None and radius is not None:
                    center_list = center.tolist() if hasattr(center, "tolist") else list(center)
                    hard_obstacle = {
                        "center":      center_list,
                        "radius":      float(radius),
                        "geometry":    geometry,
                        "avoidance":   "HARD",
                        "strength":    float(c.get("hard_strength",    0.05)),
                        "infl_factor": float(c.get("hard_infl_factor", 2.5)),
                        "projector_enabled": bool(c.get("projector_enabled", True)),
                    }
                    hard_obstacle_specs.append(hard_obstacle)
                    # Keep predicate cost geometry consistent with hard layers.
                    parameters["geometry"] = geometry

            clause = Clause(
                operator=operator,
                predicate=predicate,
                weight=weight,
                modality=modality,
                parameters=parameters,
                time_window=time_window,
                hard_obstacle=hard_obstacle,
            )

        elif operator == "until":

            left  = c["left"]
            right = c["right"]
            parameters = {
                "left_params":  extract_parameters(left,  bindings),
                "right_params": extract_parameters(right, bindings),
            }
            clause = Clause(
                operator=operator,
                predicate=(left, right),
                weight=weight,
                modality=modality,
                parameters=parameters,
                time_window=time_window,
            )

        else:
            raise ValueError(f"Unsupported operator: {operator}")

        clauses.append(clause)

    ts = TaskSpec(
        horizon_sec=horizon_sec,
        clauses=clauses,
        hard_obstacle_specs=hard_obstacle_specs,
    )
    ts.phases = phases
    return ts


def extract_parameters(predicate_name, bindings):

    params = {}
    for key, value in bindings.items():
        if key.startswith(predicate_name + "."):
            param_name = key.split(".")[1]
            if isinstance(value, list):
                value = np.array(value, dtype=float)
            params[param_name] = value
    return params
