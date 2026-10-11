"""Exercise 1: persist a trained student outcome model."""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_FILE = Path(__file__).resolve().parent / "student_pass_model.joblib"

# Illustrative dataset: [daily study hours, attendance percentage, pass label]
SAMPLES = [
    [0.5, 35, 0], [1, 45, 0], [1.5, 50, 0], [2, 55, 0],
    [2.5, 60, 0], [3, 65, 0], [3.5, 60, 0], [4, 70, 0],
    [4.5, 72, 1], [5, 75, 1], [5.5, 80, 1], [6, 78, 1],
    [6.5, 85, 1], [7, 88, 1], [8, 90, 1], [9, 95, 1],
    [2, 90, 0], [3, 95, 0], [4, 50, 0], [5, 55, 0],
    [6, 60, 1], [7, 65, 1], [8, 70, 1], [9, 75, 1],
    [1, 95, 0], [2.5, 85, 0], [3.5, 78, 0], [4.5, 65, 0],
    [5.5, 90, 1], [6.5, 70, 1], [7.5, 82, 1], [8.5, 60, 1],
]


def train():
    dataset = pd.DataFrame(
        SAMPLES,
        columns=["study_hours", "attendance_percentage", "result"]
    )
    features = dataset.loc[:, ["study_hours", "attendance_percentage"]]
    target = dataset["result"]

    classifier = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42)),
    ])
    classifier.fit(features, target)
    joblib.dump(classifier, OUTPUT_FILE)
    return classifier


def main():
    train()
    print(f"Saved trained model as {OUTPUT_FILE.name}")

    # Load the saved model and test three different input examples.
    restored = joblib.load(OUTPUT_FILE)
    test_cases = [(2, 60), (5, 80), (8, 95)]
    print("\\nExample predictions:")
    for hours, attendance in test_cases:
        row = pd.DataFrame([{
            "study_hours": hours,
            "attendance_percentage": attendance,
        }])
        label = int(restored.predict(row)[0])
        description = "Pass" if label == 1 else "Needs improvement"
        print(
            f"Study hours={hours}, attendance={attendance}% -> "
            f"{description} (class={label})"
        )


if __name__ == "__main__":
    main()
