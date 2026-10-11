# Exercise 1 - Model Persistence
import os

import joblib
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))


def path_of(filename):
    """Build a full path so the files are always saved beside this script."""
    return os.path.join(HERE, filename)


# ------------------------------------------------------------------
# Training data: [study_hours] -> 1 = Pass, 0 = Fail
# ------------------------------------------------------------------
HOURS_ONLY = [[0.5], [1], [1.5], [2], [2.5], [3], [3.5], [4], [5], [6], [7], [8], [9], [10]]
HOURS_ONLY_LABELS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

# [study_hours, attendance_percentage] -> 1 = Pass, 0 = Fail  (for Exercise 2)
HOURS_AND_ATTENDANCE = [
    [0.5, 45], [1, 55], [1.5, 60], [2, 50], [2.5, 65], [3, 72], [4, 62],
    [5, 78], [6, 82], [7, 88], [8, 92], [9, 96], [1, 98], [6, 35],
]
HOURS_AND_ATTENDANCE_LABELS = [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]


def train_and_save(features, labels, filename):
    clf = LogisticRegression()
    clf.fit(features, labels)
    joblib.dump(clf, path_of(filename))
    return clf


def main():
    # Task 1: train, then SAVE with joblib.dump() under a different filename
    train_and_save(HOURS_ONLY, HOURS_ONLY_LABELS, "student_pass_model.joblib")
    print("Task 1 - saved the model to student_pass_model.joblib")

    # Task 2: LOAD it back with joblib.load()
    model = joblib.load(path_of("student_pass_model.joblib"))
    print("Task 2 - loaded the model from student_pass_model.joblib")

    # Task 3: three predictions with different study hours
    print("Task 3 - predictions:")
    for hrs in [1, 4, 8]:
        outcome = model.predict([[hrs]])[0]
        print(f"         {hrs} hour(s) -> {'PASS' if outcome else 'FAIL'}")

    # Task 4: why retraining is not needed
    print("Task 4 - Retraining is not needed because the model already learned its")
    print("         pattern during fit(). That learned knowledge was saved in the file,")
    print("         so after joblib.load() the model can predict immediately.")

    # Model for Exercise 2 (study hours + attendance)
    train_and_save(HOURS_AND_ATTENDANCE, HOURS_AND_ATTENDANCE_LABELS, "student_pass_model_v2.joblib")
    print("Extra  - saved student_pass_model_v2.joblib (hours + attendance)")


if __name__ == "__main__":
    main()