from pathlib import Path
import joblib
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"


def train_and_save_model():
    """Train the sample model and save it to a .joblib file."""
    study_hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    results = [
        "Fail", "Fail", "Fail", "Fail", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass"
    ]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(study_hours, results)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved successfully: {MODEL_PATH.name}")
    return model


def load_saved_model():
    """Load the saved model, training it first if it does not exist."""
    if not MODEL_PATH.exists():
        print("Saved model not found. Training the model first...")
        train_and_save_model()

    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully: {MODEL_PATH.name}")
    return model


def make_prediction(model, hours):
    if hours < 0:
        raise ValueError("Study hours cannot be negative.")
    return model.predict([[hours]])[0]


def main():
    print("=" * 45)
    print("EXERCISE 1 — MODEL PERSISTENCE")
    print("=" * 45)

    model = load_saved_model()

    # Three predictions using different study-hour values.
    for hours in [2, 5, 8]:
        print(f"Study hours: {hours} -> Prediction: {make_prediction(model, hours)}")

    print("\nWhy is retraining not needed every time?")
    print(
        "The trained model is saved in a .joblib file. We can load the saved "
        "model and use it to predict new results without training it again. "
        "Retraining is needed when we want to update the model with new or "
        "changed training data."
    )


if __name__ == "__main__":
    main()
