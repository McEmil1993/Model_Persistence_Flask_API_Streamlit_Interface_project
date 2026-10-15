"""
Student Pass Prediction — Model Training (Exercises 1 & 2).

WHY THE MODEL DOES NOT NEED TO BE RETRAINED AFTER SAVING:
----------------------------------------------------------
Training (model.fit) learns numeric parameters (e.g. LogisticRegression
coefficients `coef_` and intercept `intercept_`) from the dataset. This is
expensive: it iterates over data to optimize those weights.

Serialization with `joblib.dump(model, 'student_pass_model.joblib')` writes the
*entire fitted Python estimator object* — including learned weights, feature
names, hyperparameters, and class labels — to disk as bytes (pickling).

`joblib.load('student_pass_model.joblib')` reconstructs that exact fitted
object in memory, with weights intact. Calling `model.predict()` then only
does cheap math (dot-product + sigmoid/threshold), NOT re-optimization, so
new predictions work immediately with no retraining and no access to the
original training data required.

EXERCISE 1: single-feature demo is covered by varying `study_hours` below
(2.5, 5.0, 8.5) while holding attendance at a representative 75%.
EXERCISE 2: the model natively supports BOTH `study_hours` and
`attendance_percentage` features.
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "student_pass_model.joblib")

FEATURES = ["study_hours", "attendance_percentage"]
TARGET = "passed"


def train_and_save():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    model = LogisticRegression()
    model.fit(X, y)
    # EXERCISE 1 REQUIREMENT: explicitly save under this filename
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained on {len(df)} rows, features={FEATURES}")
    print(f"Coefficients: {dict(zip(FEATURES, model.coef_[0]))}, "
          f"Intercept: {model.intercept_[0]:.4f}")
    print(f"Saved to {MODEL_PATH}")
    return model


def verify_three_predictions():
    """EXERCISE 1: reload via joblib.load and predict 3 study-hour values."""
    model = joblib.load(MODEL_PATH)
    print("\n--- EXERCISE 1 VERIFICATION (joblib.load + 3 predictions) ---")
    # Vary study_hours across required values; hold attendance at 75% (typical)
    test_cases = [
        {"study_hours": 2.5, "attendance_percentage": 75.0},
        {"study_hours": 5.0, "attendance_percentage": 75.0},
        {"study_hours": 8.5, "attendance_percentage": 75.0},
    ]
    for case in test_cases:
        X_new = pd.DataFrame([case], columns=FEATURES)
        pred = int(model.predict(X_new)[0])
        proba = float(model.predict_proba(X_new)[0][1])
        label = "Pass" if pred == 1 else "Fail"
        print(f"Study Hours={case['study_hours']}, "
              f"Attendance={case['attendance_percentage']}% "
              f"-> Prediction: {label} (P(pass)={proba:.3f})")


if __name__ == "__main__":
    train_and_save()
    verify_three_predictions()
