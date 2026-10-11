"""Train and save the student pass/fail models.

Exercise 1: saves a study-hours-only model as student_pass_model.joblib
Exercise 2: saves a study-hours + attendance model as
            student_pass_model_attendance.joblib
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_HOURS_ONLY = "student_pass_model.joblib"
MODEL_WITH_ATTENDANCE = "student_pass_model_attendance.joblib"


def make_dataset(n=400, seed=42):
    """Create a reproducible synthetic dataset of student results."""
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 10, n)
    attendance = rng.uniform(40, 100, n)
    # Passing depends on both, plus some noise.
    score = 6.0 * hours + 0.35 * attendance + rng.normal(0, 6, n)
    passed = (score >= 50).astype(int)
    return pd.DataFrame(
        {
            "study_hours": hours.round(2),
            "attendance_percentage": attendance.round(1),
            "passed": passed,
        }
    )


def train_and_save(df, features, filename):
    X = df[features]
    y = df["passed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    joblib.dump(model, filename)  # Exercise 1: save with joblib.dump()
    print(f"Saved {filename} (features={features}, test accuracy={acc:.2f})")


if __name__ == "__main__":
    data = make_dataset()
    data.to_csv("students.csv", index=False)
    train_and_save(data, ["study_hours"], MODEL_HOURS_ONLY)
    train_and_save(
        data, ["study_hours", "attendance_percentage"], MODEL_WITH_ATTENDANCE
    )
