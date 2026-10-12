"""
Exercise 1 — Model Persistence
Train a simple student pass/fail classifier and persist it with joblib.
Run: python train_model.py
"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

def train_and_save_model():
    # Example training data: study hours per day -> 0 = fail, 1 = pass.
    # Replace these sample values with your class dataset when available.
    study_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float).reshape(-1, 1)
    passed = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    model = LogisticRegression()
    model.fit(study_hours, passed)
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH}")
    return model

def main():
    model = train_and_save_model()

    # Load the persisted model and make three predictions.
    loaded_model = joblib.load(MODEL_PATH)
    for hours in (2, 5, 8):
        prediction = int(loaded_model.predict([[hours]])[0])
        label = "Pass" if prediction == 1 else "Fail"
        print(f"Study hours: {hours:>2} -> Prediction: {label}")

    print(
        "\nThe model does not need to be trained again for these predictions "
        "because joblib.load() restores the already-trained model parameters. "
        "Retraining is needed only when you want to fit it again, such as after "
        "changing or adding training data/features."
    )

if __name__ == "__main__":
    main()
