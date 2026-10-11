"""Train, save, reload, and demonstrate the student pass/fail model."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data.csv"
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"
FEATURES = ["study_hours", "attendance_percentage"]


def main():
    # Read the labeled examples and separate inputs from the target column.
    data = pd.read_csv(DATA_PATH)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(data[FEATURES], data["passed"])

    # Save the learned model so the API and interface can load it later.
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to: {MODEL_PATH}")

    # Reload from disk to demonstrate persistence before making predictions.
    loaded_model = joblib.load(MODEL_PATH)
    examples = [
        {"study_hours": 3.0, "attendance_percentage": 70},
        {"study_hours": 6.5, "attendance_percentage": 85},
        {"study_hours": 9.0, "attendance_percentage": 92},
    ]
    for example in examples:
        row = pd.DataFrame([example], columns=FEATURES)
        prediction = int(loaded_model.predict(row)[0])
        label = "Pass" if prediction == 1 else "Fail"
        print(f"Input: {example} -> prediction: {prediction} ({label})")

    print("\nThe model learned its decision pattern from labeled examples during training.")
    print("The .joblib file stores the fitted model and its learned parameters.")
    print("Loading that file restores the parameters without repeating training.")
    print("The restored model can then predict outcomes for new student inputs.")


if __name__ == "__main__":
    main()
