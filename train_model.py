from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"


def main():
    study_hours = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    results = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    model = LogisticRegression()
    model.fit(study_hours, results)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH.name}")

    loaded_model = joblib.load(MODEL_PATH)
    print("Model loaded.")

    for hours in (2, 5, 8):
        prediction = loaded_model.predict([[hours]])[0]
        label = "Pass" if prediction == 1 else "Fail"
        print(f"Study hours: {hours} -> Predicted result: {label}")


if __name__ == "__main__":
    main()
