
"""Train and save the student pass  prediction model."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def train_model():
    # Feature 1: Study hours
    # Feature 2: Attendance percentage
    X = np.array([
        [0, 40],
        [1, 50],
        [2, 55],
        [3, 60],
        [4, 65],
        [5, 70],
        [6, 80],
        [7, 85],
        [8, 90],
        [9, 92],
        [10, 95],
        [12, 100],
    ], dtype=float)

    # 0 = Needs more study, 1 = Pass
    y = np.array([
        0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1
    ])

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Number of features: {model.n_features_in_}")

    return model


if __name__ == "__main__":
    train_model()
