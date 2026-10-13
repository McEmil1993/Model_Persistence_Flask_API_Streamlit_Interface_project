from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DIR = Path(__file__).resolve().parent
HOURS_MODEL_FILE = DIR / "student_pass_model.joblib"
V2_MODEL_FILE = DIR / "student_pass_model_v2.joblib"


def new_pipeline():
    """Scale the inputs, then classify using the 3 nearest examples."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=3)),
    ])


def main():
    # ---------- training data (1 = Pass, 0 = Fail) ----------
    hours = np.array([[0.5], [1], [1.5], [2], [2.5], [3], [3.5], [4], [5], [6], [7], [8], [9], [10]])
    passed = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    pipeline = new_pipeline().fit(hours, passed)

    # Task 1: save the model with joblib.dump() using a different filename
    joblib.dump(pipeline, HOURS_MODEL_FILE)
    print("[Task 1] Model saved to student_pass_model.joblib")

    # Task 2: load the model with joblib.load()
    restored = joblib.load(HOURS_MODEL_FILE)
    print("[Task 2] Model loaded from student_pass_model.joblib")

    # Task 3: three predictions using different study hours
    print("[Task 3] Predictions:")
    for h in [1, 4, 8]:
        res = restored.predict(np.array([[h]]))[0]
        print(f"         {h} hour(s) of study -> {'PASS' if res == 1 else 'FAIL'}")

    # Task 4: explain why retraining is not needed
    print("[Task 4] No retraining is needed. The learning already happened when fit() ran,")
    print("         and what the model learned is stored in the object. joblib.dump() wrote")
    print("         that object to a file and joblib.load() read it back, so predict() can")
    print("         be used immediately.")

    # ---------- model for Exercise 2 (study hours + attendance) ----------
    both = np.array([
        [0.5, 40], [1, 50], [1.5, 60], [2, 55], [2.5, 65], [3, 70], [4, 60],
        [5, 75], [6, 80], [7, 85], [8, 90], [9, 95], [1, 95], [6, 40],
        [3.5, 72], [7, 70], [8, 60], [2, 80],
    ])
    both_passed = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0])
    joblib.dump(new_pipeline().fit(both, both_passed), V2_MODEL_FILE)
    print("[Extra]  Saved student_pass_model_v2.joblib (study hours + attendance)")


if __name__ == "__main__":
    main()