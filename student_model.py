import numpy as np


class StudentPerformanceModel:
    def fit(self, features, scores):
        x = self._validate_features(features)
        y = np.asarray(scores, dtype=float).reshape(-1)
        if len(x) != len(y) or not np.isfinite(y).all():
            raise ValueError("Provide one finite score for each training row.")
        design = np.column_stack([np.ones(len(x)), x])
        if np.linalg.matrix_rank(design) < 3:
            raise ValueError("Study hours and attendance must vary independently.")
        weights = np.linalg.lstsq(design, y, rcond=None)[0]
        self.intercept = float(weights[0])
        self.coefficients = weights[1:]
        return self

    @staticmethod
    def _validate_features(features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2 or len(x) == 0:
            raise ValueError("Provide rows of [study_hours, attendance_percentage].")
        if not np.isfinite(x).all() or (x[:, 0] < 0).any():
            raise ValueError("Inputs must be finite and study hours nonnegative.")
        if ((x[:, 1] < 0) | (x[:, 1] > 100)).any():
            raise ValueError("Attendance must be between 0 and 100.")
        return x

    def predict(self, features):
        return self.intercept + self._validate_features(features) @ self.coefficients
