from pathlib import Path
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "skilltrack_model.joblib"


def build_training_data():
    study_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 4, 6, 8, 10], dtype=float)
    attendance = np.array([30, 45, 50, 55, 60, 65, 70, 75, 80, 85, 95, 90, 85, 55, 60, 70], dtype=float)
    features = np.column_stack((study_hours, attendance))
    labels = np.array([
        "Needs Support", "Needs Support", "Needs Support", "Needs Support",
        "Developing", "Developing", "Developing", "Ready to Advance",
        "Ready to Advance", "Ready to Advance", "Ready to Advance",
        "Developing", "Ready to Advance", "Developing", "Developing",
        "Ready to Advance"
    ])
    return features, labels


def main():
    features, labels = build_training_data()
    model = DecisionTreeClassifier(max_depth=4, random_state=17)
    model.fit(features, labels)
    joblib.dump(model, MODEL_PATH)
    saved_model = joblib.load(MODEL_PATH)

    print(f"Saved model: {MODEL_PATH.name}")
    print("Three sample predictions:")
    examples = [[2, 45], [5, 65], [8, 85]]
    for study_hours, attendance in examples:
        result = saved_model.predict([[study_hours, attendance]])[0]
        print(f"Study hours={study_hours}, attendance={attendance}% -> {result}")


if __name__ == "__main__":
    main()
