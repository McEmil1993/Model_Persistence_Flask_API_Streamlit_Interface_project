"""Exercise 1: Save/load a model and predict using study hours + attendance."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")

def main():
    # Example data: [study_hours, attendance_percentage] -> 1=Pass, 0=Fail
    X = np.array([
        [0, 40], [1, 50], [2, 55], [2, 60], [3, 65],
        [4, 70], [5, 75], [6, 80], [7, 85], [8, 90],
        [9, 95], [10, 98], [11, 99], [12, 100]
    ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH.name}")

    # Load saved model and make three predictions without retraining.
    loaded_model = joblib.load(MODEL_PATH)
    examples = [[2, 55], [5, 75], [8, 90]]
    for study_hours, attendance in examples:
        predicted = int(loaded_model.predict([[study_hours, attendance]])[0])
        print(
            f"Study hours: {study_hours}, Attendance: {attendance}% -> "
            f"{'Pass' if predicted else 'Fail'}"
        )

    print(
        "\nThe model does not need training again for these predictions because "
        "joblib.load() restores the learned model parameters from the saved file."
    )

if __name__ == "__main__":
    main()
