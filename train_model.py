"""Exercise 1 - Model persistence with joblib.

Trains a Logistic Regression model that predicts whether a student passes
based on daily study hours, saves it with joblib.dump(), reloads it with
joblib.load(), and uses the reloaded model to make three predictions.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

# Training data. Feature: study hours per day. Label: 1 = pass, 0 = fail.
STUDY_HOURS = np.array(
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
     4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
).reshape(-1, 1)
PASSED = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

WHY_NO_RETRAINING = """\
Why the model does not need to be trained again to make predictions:
Training finds the model's internal parameters (here the coefficient and
intercept) that best fit the training data. joblib.dump() serialises the
already-trained estimator - its parameters, class labels and metadata - into
student_pass_model.joblib. joblib.load() restores those exact parameters into
memory. A prediction only applies the stored formula,
sigmoid(coefficient * hours + intercept), so the loaded model produces the
same results as the original without ever seeing the training data again.
Retraining is only necessary when new training data or new features are
added."""


def train_model() -> LogisticRegression:
    """Train the model and overwrite the saved one with joblib.dump()."""
    model = LogisticRegression()
    model.fit(STUDY_HOURS, PASSED)
    joblib.dump(model, MODEL_PATH)
    print(
        f"Trained model saved to '{MODEL_PATH.name}' "
        f"(coefficient={model.coef_[0][0]:.3f}, "
        f"intercept={model.intercept_[0]:.3f})."
    )
    return model


def load_model() -> LogisticRegression:
    """Load the persisted model with joblib.load()."""
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from '{MODEL_PATH.name}'.")
    return model


def main() -> None:
    train_model()
    model = load_model()

    print("\nThree predictions from the reloaded model:")
    for hours in (1.0, 4.0, 7.0):
        prediction = int(model.predict([[hours]])[0])
        print(f"  study_hours = {hours:>4} -> {'PASS' if prediction else 'FAIL'}")

    print(f"\n{WHY_NO_RETRAINING}")


if __name__ == "__main__":
    main()
