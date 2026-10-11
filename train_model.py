"""Exercise 1 - Model Persistence

Trains a pass/fail model from study hours, saves it with joblib.dump(),
loads it back with joblib.load(), and makes three predictions WITHOUT
retraining.
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"  # changed from the default filename

# ---------------------------------------------------------------------------
# 1. Training data: hours studied -> 0 = fail, 1 = pass
# ---------------------------------------------------------------------------
study_hours = np.array([[0.5], [1], [1.5], [2], [2.5], [3], [3.5], [4],
                        [4.5], [5], [5.5], [6], [7], [8], [9], [10]])
passed = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1])


def train_and_save():
    model = LogisticRegression()
    model.fit(study_hours, passed)
    joblib.dump(model, MODEL_FILE)  # Task: save with joblib.dump()
    print(f"Model trained and saved to '{MODEL_FILE}'")


def load_and_predict():
    model = joblib.load(MODEL_FILE)  # Task: load with joblib.load()
    print(f"Model loaded from '{MODEL_FILE}' (no training happens here)\n")

    # Task: three predictions with different study hours
    for hours in (1.0, 4.5, 8.0):
        label = model.predict([[hours]])[0]
        prob = model.predict_proba([[hours]])[0][1]
        result = "PASS" if label == 1 else "FAIL"
        print(f"Study hours: {hours:>4} -> {result} (pass probability {prob:.2f})")


if __name__ == "__main__":
    train_and_save()
    print()
    load_and_predict()

    print(
        "\nWhy no retraining is needed:\n"
        "Training (fit) is the step where the model learns its parameters\n"
        "(for logistic regression: the coefficient and intercept). Those learned\n"
        "values are stored inside the model object. joblib.dump() writes that\n"
        "whole fitted object to disk, and joblib.load() restores it exactly as it\n"
        "was. Predicting only applies the stored parameters to new input, so the\n"
        "training data and the fit step are not needed again."
    )
