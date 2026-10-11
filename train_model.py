from pathlib import Path

import joblib
import numpy as np

from student_model import StudentPerformanceModel


def train_and_save_model(model_path="student_pass_model.joblib"):
    # Synthetic classroom examples, not measured student outcomes.
    features = np.array([[hours, attendance]
                         for hours in [0, 2, 4, 6, 8, 10]
                         for attendance in [40, 60, 80, 100]], dtype=float)
    scores = 10 + 5 * features[:, 0] + 0.4 * features[:, 1]
    model = StudentPerformanceModel().fit(features, scores)
    joblib.dump(model, Path(__file__).resolve().parent / model_path)
    return model


def load_model(model_path="student_pass_model.joblib"):
    return joblib.load(Path(__file__).resolve().parent / model_path)


if __name__ == "__main__":
    train_and_save_model()
    model = load_model()
    print("Model trained and saved using study hours and attendance (synthetic data).")
    for attendance in [40, 80, 100]:
        score = model.predict([[5, attendance]])[0]
        print(f"Study hours: 5, attendance: {attendance}% -> predicted score: {score:.2f}")
