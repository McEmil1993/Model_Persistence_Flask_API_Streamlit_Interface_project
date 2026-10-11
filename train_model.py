"""Train and persist the illustrative student pass classifier.

This exercise uses a small, illustrative dataset because no training data was
present in the starter folder. Replace it with your real student data before
using the model for anything beyond the exercise.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"


def main() -> None:
    # One input feature: hours studied. Target values: 0 = fail, 1 = pass.
    study_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float).reshape(-1, 1)
    passed = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    model = LogisticRegression(random_state=42)
    model.fit(study_hours, passed)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    # Reload the saved artifact and use it for three different study-hour values.
    saved_model = joblib.load(MODEL_PATH)
    examples = np.array([2, 5, 8], dtype=float).reshape(-1, 1)
    predictions = saved_model.predict(examples)
    for hours, prediction in zip(examples.ravel(), predictions):
        result = "Pass" if prediction == 1 else "Fail"
        print(f"{hours:g} study hours -> {result} (class {prediction})")

    print(
        "The model does not need to be trained again for these predictions: "
        "joblib.load() restores the parameters learned during training. "
        "Call fit() again only when you want to train or update the model."
    )


if __name__ == "__main__":
    main()
