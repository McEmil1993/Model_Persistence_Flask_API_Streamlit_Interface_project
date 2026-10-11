# train_model.py - Exercise 1: Model Persistence
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "student_pass_model.joblib")
MODEL_FILE_V2 = os.path.join(BASE_DIR, "student_pass_model_v2.joblib")


def build_hours_model():
    """Train a model that uses study hours only."""
    data = pd.DataFrame({
        "hours": [0.5, 1, 1.5, 2, 2.5, 3.5, 4, 5, 6, 7, 8, 9],
        "passed": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    })
    clf = LogisticRegression()
    clf.fit(data[["hours"]].values, data["passed"].values)
    return clf


def build_attendance_model():
    """Train a model that uses study hours + attendance percentage."""
    data = pd.DataFrame({
        "hours":      [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 1, 6],
        "attendance": [40, 50, 60, 55, 65, 70, 60, 75, 80, 85, 90, 95, 95, 40],
        "passed":     [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
    })
    clf = LogisticRegression()
    clf.fit(data[["hours", "attendance"]].values, data["passed"].values)
    return clf


if __name__ == "__main__":
    # Train once, then save to a file with a different name
    trained = build_hours_model()
    joblib.dump(trained, MODEL_FILE)
    print("Model saved to: student_pass_model.joblib")

    # Load it back (no training needed anymore)
    reloaded = joblib.load(MODEL_FILE)
    print("Model loaded from: student_pass_model.joblib")

    # Three predictions with different study hours
    for h in (1, 4, 8):
        label = "PASS" if reloaded.predict([[h]])[0] == 1 else "FAIL"
        print(f"{h} hour(s) of study -> {label}")

    # Why no retraining?
    # fit() stored the learned values inside the model object.
    # dump() wrote those values to the file and load() restored them,
    # so predict() works immediately without calling fit() again.

    # Second model for Exercise 2
    joblib.dump(build_attendance_model(), MODEL_FILE_V2)
    print("Model saved to: student_pass_model_v2.joblib (hours + attendance)")