"""Train and persist the student outcome model.

Run this file once before starting the Flask or Streamlit application:
    python train_model.py
"""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"


def build_training_data():
    """Return example data with study hours and attendance percentage."""
    study_hours = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9]
    attendance = [55, 60, 72, 65, 78, 70, 82, 75, 88, 80, 92, 85, 90, 96]
    passed = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    features = [[hours, attendance_pct] for hours, attendance_pct in zip(study_hours, attendance)]
    return features, passed


def train_and_save_model():
    features, labels = build_training_data()
    model = LogisticRegression(random_state=42)
    model.fit(features, labels)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_and_save_model()
