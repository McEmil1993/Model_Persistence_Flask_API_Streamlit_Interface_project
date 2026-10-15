"""
Exercise 1 — Model Persistence
Train a simple student pass prediction model and save it with joblib.
Run: python train_model.py
"""
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

def main():
    # Example training data: study hours and whether the student passed (1) or not (0).
    # Replace these example values with your instructor's dataset if one was provided.
    study_hours = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    passed =       [0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1]

    model = LogisticRegression()
    model.fit(study_hours, passed)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH}")

    # Load the saved model and make three predictions without retraining.
    loaded_model = joblib.load(MODEL_PATH)
    for hours in [2, 5, 8]:
        prediction = int(loaded_model.predict([[hours]])[0])
        label = "Pass" if prediction == 1 else "Fail"
        print(f"Study hours: {hours} -> Prediction: {label} ({prediction})")

    print(
        "\nWhy retraining is not needed: the saved file contains the model's "
        "learned parameters. joblib.load() restores those parameters, so the "
        "model can predict new inputs directly. Retrain only when updating "
        "the model with new or changed training data."
    )

if __name__ == "__main__":
    main()
