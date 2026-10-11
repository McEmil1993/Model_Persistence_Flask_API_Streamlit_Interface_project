"""
Exercise 1 — Model Persistence
Train a simple student pass/fail model and save it for reuse.
Run: python train_model.py
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def main():
    # Small example dataset: study hours and whether the student passed (1) or failed (0).
    # Replace these sample values with your instructor's dataset if one is provided.
    study_hours = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
                            5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]).reshape(-1, 1)
    passed = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    model.fit(study_hours, passed)

    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH.name}")
    print("\nThree sample predictions (no retraining required):")
    for hours in [2, 5, 8]:
        prediction = int(model.predict([[hours]])[0])
        label = "Pass" if prediction == 1 else "Fail"
        print(f"Study hours: {hours:>4} -> Prediction: {label}")


if __name__ == "__main__":
    main()
