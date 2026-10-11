"""
Exercise 1 — Model Persistence
Train a simple student pass/fail model, save it with joblib, reload it,
and make three predictions using different study-hour values.
"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")


def build_training_data():
    # Example training data for practice purposes only.
    # In a real project, replace these examples with a reliable dataset.
    study_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
    passed = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    return study_hours, passed


def train_and_save_model():
    X, y = build_training_data()
    model = DecisionTreeClassifier(max_depth=2, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH.name}")
    return model


def main():
    train_and_save_model()

    # Reload the saved model to demonstrate persistence.
    loaded_model = joblib.load(MODEL_PATH)
    print("\nThree predictions from the loaded model:")
    for hours in [2, 5, 8]:
        prediction = int(loaded_model.predict([[hours]])[0])
        label = "Pass" if prediction == 1 else "Fail"
        print(f"Study hours: {hours:>2} -> Predicted class: {label}")

    print(
        "\nWhy retraining is not needed for these predictions:\n"
        "The trained model's learned decision rules are saved in the .joblib file. "
        "Loading that file restores the fitted model, so it can predict new inputs "
        "without learning from the training data again. Retraining is needed only "
        "when you want to change/update the model or train it on new data."
    )


if __name__ == "__main__":
    main()
