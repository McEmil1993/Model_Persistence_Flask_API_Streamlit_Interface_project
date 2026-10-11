"""Exercise 1 - Model Persistence

Trains a simple Logistic Regression that predicts whether a student passes
(1) or fails (0) from study hours only, saves it with joblib, then loads it
back and makes three predictions WITHOUT retraining.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MODEL_FILE = "student_pass_model.joblib"  # <-- new filename (Exercise 1)


def make_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 10, n)
    # The more a student studies, the more likely they pass (with some noise).
    prob = 1 / (1 + np.exp(-(hours - 5) * 1.2))
    passed = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({"study_hours": hours, "passed": passed})


def train_and_save():
    df = make_data()
    X = df[["study_hours"]]
    y = df["passed"]
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)  # save the trained model
    print(f"Model trained and saved to {MODEL_FILE}")


def load_and_predict():
    model = joblib.load(MODEL_FILE)  # load - no training happens here
    print("\nModel loaded from disk. Making predictions (no retraining):")
    for hours in [1, 5, 9]:  # three different study-hour values
        X_new = pd.DataFrame({"study_hours": [hours]})
        pred = int(model.predict(X_new)[0])
        prob = float(model.predict_proba(X_new)[0][1])
        label = "PASS" if pred == 1 else "FAIL"
        print(f"  {hours} study hours -> {label} (pass probability {prob:.2f})")


if __name__ == "__main__":
    train_and_save()
    load_and_predict()
