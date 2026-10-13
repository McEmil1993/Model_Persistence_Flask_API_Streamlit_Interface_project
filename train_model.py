"""Exercise 1 — Model Persistence (2-feature model).

Trains a classifier on [study_hours, attendance_percentage] -> pass/fail,
saves it as student_pass_model.joblib, reloads it, and makes three
predictions with different study hours.
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def build_dataset(seed: int = 42, n_samples: int = 300):
    rng = np.random.default_rng(seed)
    study_hours = rng.uniform(0, 10, n_samples)
    attendance = rng.uniform(0, 100, n_samples)
    # Pass rule: weighted combination + noise, so BOTH features matter.
    score = 0.8 * study_hours + 0.05 * attendance + rng.normal(0, 0.5, n_samples)
    labels = (score > 5.0).astype(int)
    X = np.column_stack([study_hours, attendance])
    return X, labels


def main():
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    # Pipeline scales features (hours 0-10 vs attendance 0-100) so both
    # contribute fairly; the scaler is persisted with the model.
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)
    print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

    # Task 1: Save the model using joblib.dump().
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH.name}")

    # Task 2: Load the model using joblib.load().
    loaded_model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH.name}")

    # Task 3: Make three predictions using different study hours.
    # Attendance is varied as well so both features are exercised,
    # but study hours differ in each case per the exercise requirement.
    test_cases = [
        (2.0, 60.0),
        (5.0, 75.0),
        (8.0, 90.0),
    ]
    for hours, attendance in test_cases:
        proba_pass = loaded_model.predict_proba([[hours, attendance]])[0][1]
        pred = int(proba_pass >= 0.5)
        label = "PASS" if pred == 1 else "FAIL"
        print(
            f"Study hours={hours}, Attendance={attendance}% -> {label} "
            f"(P(pass)={proba_pass:.2f})"
        )

    # Task 4: Explain why the model does not need to be trained again.
    print(
        "\nWhy no retraining is needed: joblib.dump() persists the fitted "
        "pipeline (scaler statistics plus classifier coefficients/intercept). "
        "joblib.load() restores that exact state, so predict()/predict_proba() "
        "work immediately without calling fit() again."
    )


if __name__ == "__main__":
    main()
