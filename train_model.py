from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "student_pass_model.joblib"
MODEL_V2_PATH = HERE / "student_pass_model_v2.joblib"


def make_classifier():
    return make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))


# ------------------ training data (1 = Pass, 0 = Fail) ------------------
study_hours = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
passed = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

hours_and_attendance = np.array([
    [0.5, 40], [1, 50], [1.5, 60], [2, 55], [2.5, 65], [3, 70], [4, 60],
    [5, 75], [6, 80], [7, 85], [8, 90], [9, 95], [1, 95], [6, 40],
    [3.5, 72], [7, 70], [8, 60], [2, 80],
])
passed_v2 = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0])

# ------------------ train the model ------------------
model = make_classifier()
model.fit(study_hours, passed)

joblib.dump(model, MODEL_PATH)
saved_model = joblib.load(MODEL_PATH)
for hours in (1, 4, 8):
    result = saved_model.predict([[hours]])[0]
    print(f"        {hours} hour(s) of study -> {'PASS' if result == 1 else 'FAIL'}")

model_v2 = make_classifier()
model_v2.fit(hours_and_attendance, passed_v2)
joblib.dump(model_v2, MODEL_V2_PATH)
print("Extra : saved student_pass_model_v2.joblib (study hours + attendance)")