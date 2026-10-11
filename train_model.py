"""Train and persist a student pass classifier using two input features."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "student_results.csv"
MODEL_FILE = HERE / "student_pass_model.joblib"
FEATURES = ["study_hours", "attendance_percentage"]


def predict_pass(model, study_hours, attendance_percentage):
    """Return both the binary class and its human-readable label."""
    sample = pd.DataFrame(
        [[study_hours, attendance_percentage]],
        columns=FEATURES,
    )
    result = int(model.predict(sample)[0])
    return result, "Pass" if result == 1 else "Needs support"


def main():
    records = pd.read_csv(DATA_FILE)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=17),
    )
    model.fit(records[FEATURES], records["passed"])

    # Persist the fitted pipeline, including its scaler and classifier.
    joblib.dump(model, MODEL_FILE)
    print(f"Saved trained model to {MODEL_FILE.name}")

    # Reload it to demonstrate that inference uses the saved model.
    loaded_model = joblib.load(MODEL_FILE)
    print("Three predictions from the reloaded model:")
    for hours, attendance in ((8, 60), (20, 80), (35, 95)):
        category, label = predict_pass(loaded_model, hours, attendance)
        print(
            f"  {hours:>2} study hours, {attendance}% attendance -> "
            f"{label} (class {category})"
        )

    print(
        "The model does not need training again for these predictions: joblib.load() "
        "restores the learned scaler and classifier parameters from disk. Retrain only "
        "when the training data or model design changes."
    )


if __name__ == "__main__":
    main()
