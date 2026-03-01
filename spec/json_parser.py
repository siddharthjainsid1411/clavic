import json
import numpy as np
from spec.taskspec import TaskSpec, Clause


def load_taskspec_from_json(path):

    with open(path, "r") as f:
        data = json.load(f)

    horizon_sec = data["horizon_sec"]
    bindings = data.get("bindings", {})
    phases = data.get("phases", None)       # multi-phase DMP (Scene 2+)

    clauses = []

    for c in data["clauses"]:

        operator = c["type"]
        weight = c["weight"]
        modality = c["modality"]

        # Optional time window for *_during operators
        time_window = None
        if "time_window" in c:
            time_window = tuple(c["time_window"])   # [t_start, t_end]

        # Handle unary operators
        if operator in ["always", "eventually", "always_during", "eventually_during"]:

            predicate = c["predicate"]

            parameters = extract_parameters(predicate, bindings)

            clause = Clause(
                operator=operator,
                predicate=predicate,
                weight=weight,
                modality=modality,
                parameters=parameters,
                time_window=time_window
            )

        # Handle until operator
        elif operator == "until":

            left = c["left"]
            right = c["right"]

            parameters = {
                "left_params": extract_parameters(left, bindings),
                "right_params": extract_parameters(right, bindings)
            }

            clause = Clause(
                operator=operator,
                predicate=(left, right),
                weight=weight,
                modality=modality,
                parameters=parameters,
                time_window=time_window
            )

        else:
            raise ValueError(f"Unsupported operator: {operator}")

        clauses.append(clause)

    ts = TaskSpec(
        horizon_sec=horizon_sec,
        clauses=clauses
    )
    # Attach phases if present (used by MultiPhaseCertifiedPolicy)
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