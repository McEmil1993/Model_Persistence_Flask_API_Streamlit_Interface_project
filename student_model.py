from math import isfinite
from typing import Sequence


class StudentPassModel:
    def __init__(self) -> None:
        self.decision_boundary: float | None = None

    def fit(
        self, study_hours: Sequence[Sequence[float]], outcomes: Sequence[int]
    ) -> "StudentPassModel":
        if len(study_hours) != len(outcomes) or not study_hours:
            raise ValueError(
                "Training features and outcomes must have the same nonzero length."
            )

        failures: list[float] = []
        passes: list[float] = []
        for row, outcome in zip(study_hours, outcomes):
            if len(row) != 1:
                raise ValueError("Each training row must contain one study-hours value.")
            hours = row[0]
            if isinstance(hours, bool) or not isinstance(hours, (int, float)):
                raise ValueError("Study hours must be numeric.")
            if not isfinite(hours) or hours < 0:
                raise ValueError("Study hours must be finite and zero or greater.")
            if outcome == 0:
                failures.append(float(hours))
            elif outcome == 1:
                passes.append(float(hours))
            else:
                raise ValueError("Outcomes must be 0 (fail) or 1 (pass).")

        if not failures or not passes:
            raise ValueError("Training data must include both fail and pass outcomes.")

        highest_fail_hours = max(failures)
        lowest_pass_hours = min(passes)
        if highest_fail_hours >= lowest_pass_hours:
            raise ValueError("A study-hours threshold cannot separate these outcomes.")

        self.decision_boundary = (highest_fail_hours + lowest_pass_hours) / 2
        return self

    def predict(self, study_hours: Sequence[Sequence[float]]) -> list[int]:
        if self.decision_boundary is None:
            raise ValueError("The model must be trained before making predictions.")

        predictions = []
        for row in study_hours:
            if len(row) != 1:
                raise ValueError("Each prediction row must contain one study-hours value.")
            hours = row[0]
            if isinstance(hours, bool) or not isinstance(hours, (int, float)):
                raise ValueError("Study hours must be numeric.")
            if not isfinite(hours):
                raise ValueError("Study hours must be finite.")
            predictions.append(int(hours >= self.decision_boundary))
        return predictions
