from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).resolve().parent / "learnlens_model.joblib"


def main():
    study_hours = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 4, 6, 8, 10], dtype=float)
    attendance = np.array([40, 55, 60, 50, 65, 70, 75, 80, 85, 90, 95, 90, 85, 60, 70, 80], dtype=float)

    labels = []
    for hours, rate in zip(study_hours, attendance):
        score = hours * 6 + rate * 0.35
        if score < 35:
            labels.append("Needs Support")
        elif score < 55:
            labels.append("On Track")
        else:
            labels.append("Strong Progress")

    features = np.column_stack((study_hours, attendance))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(features, labels)

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH.name}")

    saved_model = joblib.load(MODEL_PATH)
    print("Loaded saved model successfully.")

    samples = [(2, 55), (5, 75), (8, 90)]
    for hours, rate in samples:
        result = saved_model.predict([[hours, rate]])[0]
        print(f"Study hours: {hours}, attendance: {rate}% -> {result}")


if __name__ == "__main__":
    main()
