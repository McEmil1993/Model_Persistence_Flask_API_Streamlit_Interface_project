from pathlib import Path
import joblib
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "studymate_model.joblib"


def main():
    features = [
        [0, 30], [1, 40], [2, 45], [2, 70], [3, 55],
        [3, 80], [4, 60], [4, 85], [5, 65], [5, 90],
        [6, 70], [6, 95], [7, 75], [7, 98], [8, 80],
        [8, 100], [9, 85], [10, 95]
    ]
    outcomes = [
        "Needs Support", "Needs Support", "Needs Support", "Needs Support",
        "Needs Support", "Needs Support", "Needs Support", "On Track",
        "Needs Support", "On Track", "On Track", "On Track",
        "On Track", "On Track", "On Track", "On Track",
        "On Track", "On Track"
    ]

    model = DecisionTreeClassifier(max_depth=3, random_state=7)
    model.fit(features, outcomes)
    joblib.dump(model, MODEL_FILE)
    print(f"Saved model: {MODEL_FILE.name}")

    saved_model = joblib.load(MODEL_FILE)
    print("Loaded saved model successfully.")
    for hours, attendance in [[2, 45], [5, 90], [8, 80]]:
        result = saved_model.predict([[hours, attendance]])[0]
        print(f"Study hours: {hours}, attendance: {attendance}% -> {result}")


if __name__ == "__main__":
    main()
