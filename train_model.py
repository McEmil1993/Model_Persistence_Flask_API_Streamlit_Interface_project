from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

FOLDER = Path(__file__).resolve().parent


class ModelStore:
    """Small helper that trains, saves and loads a model file."""

    def __init__(self, filename):
        self.path = FOLDER / filename

    def train(self, features, labels):
        # random_state makes the result identical every time
        clf = GradientBoostingClassifier(n_estimators=50, random_state=0)
        return clf.fit(features, labels)

    def save(self, model):
        joblib.dump(model, self.path)          # Task 1

    def load(self):
        return joblib.load(self.path)          # Task 2


hours = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

hours_attendance = np.array([
    [0.5, 40], [1, 50], [1.5, 60], [2, 55], [2.5, 65], [3, 70], [4, 60],
    [5, 75], [6, 80], [7, 85], [8, 90], [9, 95], [1, 95], [6, 40],
    [3.5, 72], [7, 70], [8, 60], [2, 80],
])
labels_v2 = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0])

store = ModelStore("student_pass_model.joblib")

store.save(store.train(hours, labels))
print("Task 1 -> model saved to student_pass_model.joblib")

model = store.load()
print("Task 2 -> model loaded from student_pass_model.joblib")

print("Task 3 -> predictions:")
for h in (1, 4, 8):
    outcome = model.predict(np.array([[h]]))[0]
    print(f"          {h} hour(s) of study = {'PASS' if outcome == 1 else 'FAIL'}")

print("Task 4 -> The model learned its patterns during fit() and keeps them inside the")
print("          model object. joblib.dump() saved that object and joblib.load() restored")
print("          it, so it can predict right away without being trained again.")

store_v2 = ModelStore("student_pass_model_v2.joblib")
store_v2.save(store_v2.train(hours_attendance, labels_v2))
print("Extra  -> saved student_pass_model_v2.joblib (study hours + attendance)")