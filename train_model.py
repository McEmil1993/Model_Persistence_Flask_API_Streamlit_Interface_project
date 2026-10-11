"""
Student Pass Prediction — Logistic Regression Model
Trains a simple model on study hours vs. pass/fail, saves it, reloads it, and tests predictions.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "student_pass_model.joblib"


def train_model(X, y):
    """Train a LogisticRegression model on the given data."""
    model = LogisticRegression()
    model.fit(X, y)
    return model


def save_model(model, path=MODEL_PATH):
    """Save the trained model to disk."""
    joblib.dump(model, path)
    print(f"[OK] Model saved to '{path}'")


def load_model(path=MODEL_PATH):
    """Load a saved model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model file found at '{path}'")
    model = joblib.load(path)
    print(f"[OK] Model loaded from '{path}'")
    return model


def predict(model, hours):
    """Return 'Pass' or 'Fail' for a given number of study hours."""
    prob = model.predict_proba([[hours]])[0][1]   # probability of passing
    label = "Pass" if prob >= 0.5 else "Fail"
    return label, prob


def main():
    # --- Training data ---
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # --- Train ---
    model = train_model(X, y)

    # --- Save ---
    save_model(model)

    # --- Reload (to verify persistence works) ---
    loaded_model = load_model()

    # --- Test predictions ---
    print("\nPredictions:")
    print(f"{'Hours':>6} | {'Result':<6} | {'Probability':>11}")
    print("-" * 32)
    for hours in [2, 5, 8]:
        result, prob = predict(loaded_model, hours)
        print(f"{hours:>6} | {result:<6} | {prob:>10.2%}")


if __name__ == "__main__":
    main()