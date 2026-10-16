"""
Student Pass Prediction - Model Training and Persistence
Exercise 1: Save, load, and reuse a trained model.
"""

from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"


def main():
    # Example training data: study hours and outcome (0 = Fail, 1 = Pass).
    # Replace these example values with your instructor's dataset if provided.
    study_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    results = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    model = LogisticRegression()
    model.fit(study_hours, results)

    # Save the trained model to a file.
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH.name}")

    # Load the saved model. It does not need to be trained again for predictions.
    loaded_model = joblib.load(MODEL_PATH)

    print("\nThree sample predictions:")
    for hours in [2, 5, 8]:
        prediction = int(loaded_model.predict([[hours]])[0])
        label = "Pass" if prediction == 1 else "Fail"
        print(f"Study hours: {hours} -> Predicted class: {label} ({prediction})")

    print(
        "\nExplanation: joblib.load() restores the trained model, including its "
        "learned parameters. We can call predict() without fitting the model again. "
        "Retraining is needed only when we want to learn from new or changed training data."
    )


if __name__ == "__main__":
    main()
