"""Train and persist the student pass/fail model.

Run this script once before starting the Flask or Streamlit app. Predictions
can then be made by loading the saved model; the model does not need to be
trained again for each student.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"


def main():
    data = {
        "study_hours": [
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0,
            6.5, 7.0, 8.0, 8.5, 9.0,
        ],
        "attendance_percentage": [
            50, 55, 60, 65, 70, 60, 75, 80, 70, 85,
            90, 85, 90, 95, 98,
        ],
        "passed": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    df = pd.DataFrame(data)
    features = df[["study_hours", "attendance_percentage"]]
    target = df["passed"]

    model = LogisticRegression(random_state=42)
    model.fit(features, target)

    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH}")

    # Load the persisted model and use it for three new predictions.
    saved_model = joblib.load(MODEL_PATH)
    examples = pd.DataFrame(
        {
            "study_hours": [2.0, 5.0, 8.0],
            "attendance_percentage": [60, 80, 95],
        }
    )
    predictions = saved_model.predict(examples)
    for row, prediction in zip(examples.to_dict(orient="records"), predictions):
        result = "Pass" if int(prediction) == 1 else "Fail"
        print(
            f"Study hours: {row['study_hours']:.1f}, "
            f"attendance: {row['attendance_percentage']}% -> {result}"
        )

    print(
        "\nThe saved model contains the learned parameters from training. "
        "Loading it lets the apps make predictions without fitting it again."
    )


if __name__ == "__main__":
    main()
