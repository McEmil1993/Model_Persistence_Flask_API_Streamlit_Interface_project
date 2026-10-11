"""
Exercise 1 - Model Persistence

Trains the "will the student pass?" model, saves it to disk with joblib,
loads it back, and makes three predictions without retraining.

Two models are saved:

1. student_pass_model.joblib
   Uses study hours only. This is the model for Exercise 1 and for the
   Streamlit app in Exercise 3.

2. student_pass_attendance_model.joblib
   Uses study hours AND attendance percentage. This is the model used by the
   Flask API in Exercise 2, so attendance really influences the prediction.

Run:  python train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent

# Exercise 1: "a different filename, such as student_pass_model.joblib"
HOURS_MODEL_PATH = BASE_DIR / "student_pass_model.joblib"
# Exercise 2: model that also uses attendance
ATTENDANCE_MODEL_PATH = BASE_DIR / "student_pass_attendance_model.joblib"

# Small hand-written dataset: (study_hours, attendance_percentage, passed)
# A few "surprising" rows are included on purpose so the data is not
# perfectly separable (real data never is).
DATA = [
    (0.5, 55, 0),
    (1.0, 60, 0),
    (1.5, 70, 0),
    (2.0, 50, 0),
    (2.0, 85, 0),
    (2.5, 65, 0),
    (3.0, 60, 0),
    (3.0, 90, 0),
    (3.5, 70, 0),
    (4.0, 55, 0),
    (4.0, 80, 1),
    (4.5, 75, 0),
    (5.0, 60, 0),
    (5.0, 85, 1),
    (5.5, 70, 1),
    (6.0, 65, 1),
    (6.0, 50, 0),
    (6.5, 90, 1),
    (7.0, 75, 1),
    (7.5, 80, 1),
    (8.0, 60, 1),
    (8.5, 95, 1),
    (9.0, 70, 1),
    (10.0, 85, 1),
]


def load_data() -> pd.DataFrame:
    return pd.DataFrame(DATA, columns=["study_hours", "attendance_percentage", "passed"])


def train_models(df: pd.DataFrame):
    """Return (hours_only_model, hours_and_attendance_model)."""
    y = df["passed"]

    hours_model = LogisticRegression(max_iter=1000)
    hours_model.fit(df[["study_hours"]], y)

    attendance_model = LogisticRegression(max_iter=1000)
    attendance_model.fit(df[["study_hours", "attendance_percentage"]], y)

    return hours_model, attendance_model


def main() -> None:
    df = load_data()
    hours_model, attendance_model = train_models(df)

    # --- Task 1: save the model using joblib.dump() -------------------------
    joblib.dump(hours_model, HOURS_MODEL_PATH)
    joblib.dump(attendance_model, ATTENDANCE_MODEL_PATH)
    print(f"Saved: {HOURS_MODEL_PATH.name}")
    print(f"Saved: {ATTENDANCE_MODEL_PATH.name}")

    # --- Task 2: load the model using joblib.load() -------------------------
    # From here on we only use `loaded_model`, the copy read back from disk.
    loaded_model = joblib.load(HOURS_MODEL_PATH)
    print(f"\nLoaded {HOURS_MODEL_PATH.name} from disk (no training happens here).")

    # --- Task 3: three predictions using different study hours --------------
    print("\nPredictions from the loaded model:")
    for hours in (1.5, 5.0, 8.0):
        row = pd.DataFrame({"study_hours": [hours]})
        predicted_class = int(loaded_model.predict(row)[0])
        pass_probability = float(loaded_model.predict_proba(row)[0][1])
        label = "PASS" if predicted_class == 1 else "FAIL"
        print(f"  {hours:>4} study hours -> {label}  (pass probability {pass_probability:.2f})")

    # --- Task 4: why no retraining is needed --------------------------------
    print(
        "\nWhy no retraining was needed:\n"
        "  Training (model.fit) is the slow step where the model learns its\n"
        "  parameters (the coefficient and intercept) from the data. joblib.dump()\n"
        "  stores those learned parameters in the file. joblib.load() restores\n"
        "  them, so the loaded model is already trained. Predicting is just a\n"
        "  quick calculation with those stored numbers, so we can reuse the same\n"
        "  model any number of times (and in other programs such as the Flask API\n"
        "  or the Streamlit app) without ever calling fit() again."
    )


if __name__ == "__main__":
    main()