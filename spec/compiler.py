# spec/compiler.py

from logic import predicates
from logic import temporal_logic


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

                # Slack variable s >= 0
                s = max(0.0, -rho)

                # Quadratic slack penalty
                total_cost += SLACK_WEIGHT * (s ** 2)

            # Soft clauses contribute to cost
            for clause in soft_clauses:
                rho = self._evaluate_clause(trace, clause)
                J = max(0.0, -rho)
                total_cost += clause.weight * J

            return total_cost

        return objective

    def _evaluate_clause(self, trace, clause):

        predicate_fn = self.predicate_registry[clause.predicate]
        rho_trace = predicate_fn(trace, **clause.parameters)

        if clause.operator == "eventually":
            return temporal_logic.eventually(rho_trace)

        elif clause.operator == "always":
            return temporal_logic.always(rho_trace)

        else:
            raise NotImplementedError(
                f"Operator {clause.operator} not supported yet."
            )