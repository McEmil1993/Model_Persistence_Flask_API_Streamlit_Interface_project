# Exercise 1 - Model Persistence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

OUTPUT_DIR = Path(__file__).parent.resolve()
MAIN_MODEL = OUTPUT_DIR / "student_pass_model.joblib"        # different filename (Task 1)
ATTENDANCE_MODEL = OUTPUT_DIR / "student_pass_model_v2.joblib"  # for Exercise 2


@dataclass
class Dataset:
    features: np.ndarray
    labels: np.ndarray


# Study hours only (1 = Pass, 0 = Fail)
hours_data = Dataset(
    features=np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]).reshape(-1, 1),
    labels=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
)

# Study hours + attendance percentage (for Exercise 2)
attendance_data = Dataset(
    features=np.array([
        [0.5, 42], [1, 52], [1.5, 58], [2, 48], [2.5, 66], [3, 74], [4, 63],
        [5, 77], [6, 83], [7, 87], [8, 91], [9, 97], [1, 99], [6, 30],
    ]),
    labels=np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]),
)


def fit_model(data: Dataset) -> LogisticRegression:
    clf = LogisticRegression()
    return clf.fit(data.features, data.labels)


# ---- Train, then Task 1: save with joblib.dump() ----
trained_model = fit_model(hours_data)
joblib.dump(trained_model, MAIN_MODEL)
print(f"1) Saved model  -> {MAIN_MODEL.name}")

# ---- Task 2: load with joblib.load() ----
model = joblib.load(MAIN_MODEL)
print(f"2) Loaded model <- {MAIN_MODEL.name}")

# ---- Task 3: three predictions with different study hours ----
print("3) Predictions from the loaded model:")
for h in (1, 4, 8):
    verdict = "PASS" if model.predict(np.array([[h]]))[0] == 1 else "FAIL"
    print(f"   {h} hour(s) of study -> {verdict}")

# ---- Task 4: why no retraining ----
print("4) No retraining needed: fit() already stored what the model learned, joblib.dump()")
print("   wrote it to the file, and joblib.load() restored it, so predict() works right away.")

# ---- Extra model for Exercise 2 ----
joblib.dump(fit_model(attendance_data), ATTENDANCE_MODEL)
print(f"5) Saved model  -> {ATTENDANCE_MODEL.name} (study hours + attendance)")