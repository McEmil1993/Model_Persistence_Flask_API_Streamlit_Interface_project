import math
from collections.abc import Sequence


EXAMPLE_STUDY_HOURS = [[hours] for hours in range(11)]
EXAMPLE_PASS_LABELS = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


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
        return [int(probabilities[1] >= 0.5) for probabilities in self.predict_proba(study_hours)]

    def predict_proba(self, study_hours: Sequence[Sequence[float]]) -> list[list[float]]:
        probabilities = [self._probability(self.weight * features[0] + self.bias) for features in study_hours]
        return [[1 - probability, probability] for probability in probabilities]

    @staticmethod
    def _probability(logit: float) -> float:
        if logit >= 0:
            return 1 / (1 + math.exp(-logit))
        exponential = math.exp(logit)
        return exponential / (1 + exponential)


def leave_one_out_metrics(
    study_hours: Sequence[Sequence[float]], passed: Sequence[int]
) -> tuple[float, float]:
    if len(study_hours) != len(passed) or len(passed) < 2:
        raise ValueError("Evaluation requires matching features and labels for at least two examples.")

    correct_predictions = 0
    total_squared_probability_error = 0.0

    for held_out_index, (held_out_features, actual_label) in enumerate(zip(study_hours, passed)):
        training_features = [features for index, features in enumerate(study_hours) if index != held_out_index]
        training_labels = [label for index, label in enumerate(passed) if index != held_out_index]

        evaluation_model = StudentPassModel()
        evaluation_model.fit(training_features, training_labels)
        pass_probability = evaluation_model.predict_proba([held_out_features])[0][1]
        predicted_label = int(pass_probability >= 0.5)

        correct_predictions += int(predicted_label == actual_label)
        total_squared_probability_error += (pass_probability - actual_label) ** 2

    example_count = len(passed)
    return correct_predictions / example_count, total_squared_probability_error / example_count