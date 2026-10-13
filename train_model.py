"""Train, save, load, and use a simple student pass prediction model."""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def train_and_save_model():
    # Example training data: weekly study hours and whether the student passed.
    study_hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    passed = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    model = LogisticRegression(random_state=42)
    model.fit(study_hours, passed)
    joblib.dump(model, MODEL_PATH)
    return model


if __name__ == "__main__":
    train_and_save_model()
    print(f"Model saved to: {MODEL_PATH.name}")

    # Loading the saved model means training is not needed again for predictions.
    saved_model = joblib.load(MODEL_PATH)
    for hours in (2, 5, 8):
        prediction = saved_model.predict([[hours]])[0]
        result = "Pass" if prediction == 1 else "Fail"
        print(f"Study hours: {hours} -> Predicted class: {result}")

    print("The fitted model is stored in the .joblib file, so it can make new predictions without retraining.")
