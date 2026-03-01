# spec/compiler.py

from logic import predicates
from logic import temporal_logic
import numpy as np


class Compiler:

    def __init__(self, predicate_registry):
        self.predicate_registry = predicate_registry

    def compile(self, taskspec):

        hard_clauses = []
        soft_clauses = []

        for clause in taskspec.clauses:
            if clause.modality == "REQUIRE":
                hard_clauses.append(clause)
            else:
                soft_clauses.append(clause)

        def objective(trace):

            total_cost = 0.0

            # --- Hard clauses with slack relaxation ---
            SLACK_WEIGHT = 500.0  # λ_s (tuneable)

            for clause in hard_clauses:
                rho = self._evaluate_clause(trace, clause)
                s = max(0.0, -rho)
                total_cost += SLACK_WEIGHT * (s ** 2)

            # --- Soft clauses ---
            for clause in soft_clauses:
                rho = self._evaluate_clause(trace, clause)
                J = max(0.0, -rho)
                total_cost += clause.weight * J

            # --- Intrinsic stiffness regularizer (no JSON entry needed) ---
            # Two-layer defence:
            #   1. set_theta() clips SK/SD weights to ±SK_CLIP (Fix B) → ODE stays bounded
            #   2. This penalty acts on the RAW (pre-clip) theta stored on the trace,
            #      so PI2 sees a real cost for samples with large SK/SD weights even
            #      when the clip absorbs them.  This steers the PI2 mean away from the
            #      clip boundary honestly, preventing mean drift.
            #
            # Penalty form: λ · mean(max(0, |w| - SK_CLIP)²)
            #   - mean (not sum) → scale independent of n_bfs_slack
            #   - zero inside ±SK_CLIP, grows quadratically outside
            #   - at SK=±20 (5 over clip): penalty = 1.0 * 25 / 42 ≈ 0.6 (small, informative)
            #   - at SK=±50 (35 over clip): penalty = 1.0 * 1225 / 42 ≈ 29  (large, PI2 avoids)
            SK_CLIP      = 15.0
            STIFF_WEIGHT = 1.0
            if hasattr(trace, 'raw_sk_weights') and trace.raw_sk_weights is not None:
                w = trace.raw_sk_weights
                excess = np.maximum(0.0, np.abs(w) - SK_CLIP)
                total_cost += STIFF_WEIGHT * float(np.mean(excess**2))
            if hasattr(trace, 'raw_sd_weights') and trace.raw_sd_weights is not None:
                w = trace.raw_sd_weights
                excess = np.maximum(0.0, np.abs(w) - SK_CLIP)
                total_cost += STIFF_WEIGHT * float(np.mean(excess**2))

            return total_cost

        return objective

    def _evaluate_clause(self, trace, clause):

        predicate_fn = self.predicate_registry[clause.predicate]
        rho_trace = predicate_fn(trace, **clause.parameters)

        if clause.operator == "eventually":
            return temporal_logic.eventually(rho_trace)

        elif clause.operator == "always":
            return temporal_logic.always(rho_trace)

        elif clause.operator == "always_during":
            t_start = clause.time_window[0]
            t_end   = clause.time_window[1]
            return temporal_logic.always_during(rho_trace, trace.time, t_start, t_end)

        elif clause.operator == "eventually_during":
            t_start = clause.time_window[0]
            t_end   = clause.time_window[1]
            return temporal_logic.eventually_during(rho_trace, trace.time, t_start, t_end)

        elif clause.operator == "until":
            # Until expects two predicates: (left, right)
            left_fn  = self.predicate_registry[clause.predicate[0]]
            right_fn = self.predicate_registry[clause.predicate[1]]
            rho_phi  = left_fn(trace, **clause.parameters["left_params"])
            rho_psi  = right_fn(trace, **clause.parameters["right_params"])
            return temporal_logic.until(rho_phi, rho_psi)

        else:
            raise NotImplementedError(
                f"Operator {clause.operator} not supported yet."
            )