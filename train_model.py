"""Train, save, load, and use a simple student pass prediction model."""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression


MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def train_and_save_model():
    """Train a model that predicts whether a student passes from study hours."""
    # Each inner list is one student's study hours; 0 means fail and 1 means pass.
    study_hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    passed = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    model = LogisticRegression(random_state=42)
    model.fit(study_hours, passed)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    """Load the saved model, training it first if it does not exist yet."""
    if not MODEL_PATH.exists():
        train_and_save_model()
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train_and_save_model()
    saved_model = load_model()

    print(f"Model saved to: {MODEL_PATH.name}")
    for hours in (2, 5, 8):
        prediction = saved_model.predict([[hours]])[0]
        result = "Pass" if prediction == 1 else "Fail"
        print(f"{hours} study hours: {result}")

    print(
        "\nThe model does not need to be trained again because joblib.load() "
        "restores the already-trained model from the saved file."
    )
