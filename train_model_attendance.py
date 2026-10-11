"""Exercise 2 (retraining step) - model that uses study hours AND attendance.

Run this once before starting the Flask API (app.py).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model_attendance.joblib"


def make_data(n=300, seed=7):
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 10, n)
    attendance = rng.uniform(40, 100, n)
    score = (hours - 5) * 0.9 + (attendance - 70) * 0.06
    prob = 1 / (1 + np.exp(-score))
    passed = (rng.random(n) < prob).astype(int)
    return pd.DataFrame(
        {"study_hours": hours, "attendance_percentage": attendance, "passed": passed}
    )


if __name__ == "__main__":
    df = make_data()
    X = df[["study_hours", "attendance_percentage"]]
    y = df["passed"]
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print(f"Saved {MODEL_FILE} (accuracy on training data: {model.score(X, y):.2f})")
