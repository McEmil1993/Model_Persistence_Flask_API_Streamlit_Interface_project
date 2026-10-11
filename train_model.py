"""Train and persist the student pass classifier, then demonstrate loading it."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "students.csv"
MODEL_PATH = BASE_DIR / "student_pass_model.joblib"


def train_and_save_model() -> LogisticRegression:
    """Train from the sample CSV and save the fitted model to disk."""
    data = pd.read_csv(DATA_PATH)
    model = LogisticRegression()
    model.fit(data[["study_hours", "attendance_percentage"]], data["passed"])
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH.name}")
    return model


def main() -> None:
    train_and_save_model()

    # Load the saved model instead of fitting it again.
    saved_model = joblib.load(MODEL_PATH)
    example_students = pd.DataFrame(
        {
            "study_hours": [2.0, 5.0, 8.0],
            "attendance_percentage": [60.0, 78.0, 95.0],
        }
    )
    predictions = saved_model.predict(example_students)

    print("Predictions from the saved model:")
    for hours, attendance, prediction in zip(
        example_students["study_hours"],
        example_students["attendance_percentage"],
        predictions,
    ):
        label = "Pass" if prediction == 1 else "Fail"
        print(f"  {hours:g} study hours, {attendance:g}% attendance -> {label}")

    print(
        "\nThe fitted model parameters are stored in the joblib file. "
        "Loading restores those parameters, so prediction does not require "
        "training again."
    )


if __name__ == "__main__":
    main()
