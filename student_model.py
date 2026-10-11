import math
from collections.abc import Sequence


class StudentPassModel:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 5000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight = 0.0
        self.bias = 0.0

    def fit(self, study_hours: Sequence[Sequence[float]], passed: Sequence[int]) -> None:
        sample_count = len(study_hours)
        for _ in range(self.epochs):
            weight_gradient = 0.0
            bias_gradient = 0.0
            for features, label in zip(study_hours, passed):
                predicted_probability = self._probability(self.weight * features[0] + self.bias)
                error = predicted_probability - label
                weight_gradient += error * features[0]
                bias_gradient += error

            self.weight -= self.learning_rate * weight_gradient / sample_count
            self.bias -= self.learning_rate * bias_gradient / sample_count

    def predict(self, study_hours: Sequence[Sequence[float]]) -> list[int]:
        return [int(self._probability(self.weight * features[0] + self.bias) >= 0.5) for features in study_hours]

    @staticmethod
    def _probability(logit: float) -> float:
        if logit >= 0:
            return 1 / (1 + math.exp(-logit))
        exponential = math.exp(logit)
        return exponential / (1 + exponential)