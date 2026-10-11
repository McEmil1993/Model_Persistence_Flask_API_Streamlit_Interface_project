import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Illustrative training records with both requested inputs.
hours = [1, 2, 3, 4, 5, 6, 7, 8, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 0.5, 2, 4, 6, 8, 1, 3, 5, 7, 9, 0, 3, 5, 7]
attendance = [55, 60, 65, 70, 75, 80, 85, 95, 80, 50, 90, 60, 88, 75, 92, 98, 40, 95, 55, 90, 70, 100, 85, 70, 95, 90, 20, 60, 100, 80]
passed = [0,0,0,0,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,1,1,0,1,0,1,1,0,0,1,1]
data = pd.DataFrame({"study_hours": hours, "attendance_percentage": attendance, "passed": passed})

def main():
    features = data[["study_hours", "attendance_percentage"]]
    target = data["passed"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=17, stratify=target
    )
    model = RandomForestClassifier(n_estimators=120, max_depth=4, random_state=17)
    model.fit(x_train, y_train)
    print(f"Holdout accuracy: {accuracy_score(y_test, model.predict(x_test)):.2f}")

    model_path = "student_pass_model.joblib"
    joblib.dump(model, model_path)
    restored_model = joblib.load(model_path)

    examples = pd.DataFrame({
        "study_hours": [2, 5, 8],
        "attendance_percentage": [60, 75, 95],
    })
    outcomes = restored_model.predict(examples)
    for row, prediction in zip(examples.itertuples(index=False), outcomes):
        label = "Pass" if prediction == 1 else "Needs improvement"
        print(f"{row.study_hours} hours, {row.attendance_percentage}% attendance -> {label}")

if __name__ == "__main__":
    main()
