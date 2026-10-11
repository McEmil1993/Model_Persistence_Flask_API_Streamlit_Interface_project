from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).parent / "student_pass_model.joblib"
FEATURES = ["study_hours", "attendance_percentage"]


def build_dataset(n_students: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    study_hours = rng.uniform(0, 10, n_students)
    attendance = rng.uniform(40, 100, n_students)

    score = 0.6 * study_hours + 0.04 * attendance + rng.normal(0, 0.6, n_students)
    passed = (score > 5.8).astype(int)

    return pd.DataFrame(
        {
            "study_hours": study_hours.round(1),
            "attendance_percentage": attendance.round(1),
            "passed": passed,
        }
    )


def main() -> None:
    data = build_dataset()
    X = data[FEATURES]
    y = data["passed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)

    print(f"Training accuracy: {model.score(X_train, y_train):.2f}")
    print(f"Test accuracy:     {model.score(X_test, y_test):.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()