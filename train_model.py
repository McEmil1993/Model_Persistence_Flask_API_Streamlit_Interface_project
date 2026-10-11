# Exercise 1
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FOLDER = Path(__file__).resolve().parent
SAVE_AS = FOLDER / "student_pass_model.joblib"        # Task 1: different filename
SAVE_AS_V2 = FOLDER / "student_pass_model_v2.joblib"  # used by Exercise 2

hours = np.array([0.5, 1, 1.5, 2, 2.5, 3.5, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
results = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# Train the model (hours only)
hours_model = make_pipeline(StandardScaler(), LogisticRegression())
hours_model.fit(hours, results)

joblib.dump(hours_model, SAVE_AS)
print(f"[1] Model saved as: {SAVE_AS.name}")
saved_model = joblib.load(SAVE_AS)
print(f"[2] Model loaded from: {SAVE_AS.name}")
print("[3] Predictions:")
for study_hours in [1, 4, 8]:
    answer = saved_model.predict([[study_hours]])[0]
    print(f"    {study_hours} hour(s) of study -> {'PASS' if answer == 1 else 'FAIL'}")
print("[4] Why no retraining? The patterns learned during fit() are stored inside")
print("    the model. joblib.dump() saved them to a file and joblib.load() brought")
print("    them back, so predict() can answer right away without calling fit() again.")
both = np.array([
    [0.5, 40], [1, 50], [1.5, 60], [2, 55], [2.5, 65], [3, 70], [4, 60],
    [5, 75], [6, 80], [7, 85], [8, 90], [9, 95], [1, 95], [6, 40],
])
both_results = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0])

both_model = make_pipeline(StandardScaler(), LogisticRegression())
both_model.fit(both, both_results)
joblib.dump(both_model, SAVE_AS_V2)
print(f"[5] Second model saved as: {SAVE_AS_V2.name} (hours + attendance)")