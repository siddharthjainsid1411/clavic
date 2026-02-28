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

            # --- Stiffness rate penalty (penalises steep jumps in tr(K)) ---
            # The Q ODE is untouched — this is purely a PI2 cost signal.
            # We penalise large changes in tr(K) between consecutive timesteps,
            # which physically correspond to sudden torque demands on the robot.
            #
            # Implementation: RMS of d(tr(K))/dt along the trajectory.
            # Normalised by K0 so the penalty is dimensionless.
            #   smooth traj (rms ~200 N/m/s)  → penalty ≈ 0.1   (small)
            #   spiky  traj (rms ~5000 N/m/s) → penalty ≈ 2.5   (meaningful)
            RATE_WEIGHT = 5e-4
            K0_norm     = 200.0   # nominal stiffness for normalisation
            if hasattr(trace, 'gains') and trace.gains is not None:
                K_arr = trace.gains["K"]
                trK   = np.array([np.trace(K_arr[i]) for i in range(len(K_arr))])
                dt    = float(trace.time[1] - trace.time[0]) if len(trace.time) > 1 else 0.01
                dtrK  = np.diff(trK) / dt
                total_cost += RATE_WEIGHT * float(np.sqrt(np.mean(dtrK**2)))

            return total_cost

        return objective

    def _evaluate_clause(self, trace, clause):

        predicate_fn = self.predicate_registry[clause.predicate]

        # --- Deadline slicing ---
        # If the clause has a deadline_sec, evaluate the predicate only on the
        # portion of the trajectory up to that time.  The DMP always runs for
        # the full horizon_sec; the deadline just restricts which part the
        # temporal logic operator sees.
        #
        # Example: AtGoalPose with deadline_sec=1.5 on a horizon_sec=2.0 traj
        # means "eventually reach goal within the first 1.5s".
        #
        # No deadline → use full trace (existing behaviour, backward-compatible).
        if clause.deadline_sec is not None:
            import types
            t = trace.time
            mask = t <= clause.deadline_sec
            # Build a lightweight view — only slice numpy arrays
            sliced = types.SimpleNamespace()
            sliced.time     = t[mask]
            sliced.position = trace.position[mask]
            sliced.velocity = trace.velocity[mask] if trace.velocity is not None else None
            sliced.gains    = {k: v[mask] for k, v in trace.gains.items()} if trace.gains else None
            sliced.raw_sk_weights = trace.raw_sk_weights
            sliced.raw_sd_weights = trace.raw_sd_weights
            eval_trace = sliced
        else:
            eval_trace = trace

        rho_trace = predicate_fn(eval_trace, **clause.parameters)

        if clause.operator == "eventually":
            return temporal_logic.eventually(rho_trace)

        elif clause.operator == "always":
            return temporal_logic.always(rho_trace)

        else:
            raise NotImplementedError(
                f"Operator {clause.operator} not supported yet."
            )