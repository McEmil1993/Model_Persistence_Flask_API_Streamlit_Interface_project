import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("students.csv")
y = data["passed"]

# EXERCISE 1 - Model persistence (study hours only)
X_hours = data[["study_hours"]]
X_train, X_test, y_train, y_test = train_test_split(
    X_hours, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Study-hours model accuracy on test data: {accuracy_score(y_test, model.predict(X_test)):.2f}")

joblib.dump(model, "student_pass_model.joblib")
print("Saved: student_pass_model.joblib")

loaded_model = joblib.load("student_pass_model.joblib")

for hours in [2, 5, 9]:
    new_student = pd.DataFrame({"study_hours": [hours]})
    pred = loaded_model.predict(new_student)[0]
    prob = loaded_model.predict_proba(new_student)[0][1]
    print(f"Study hours = {hours}: {'Pass' if pred == 1 else 'Fail'} (pass probability {prob:.2f})")

# Why no retraining? fit() already stored the learned values (coefficient
# and intercept) inside the model object. joblib.dump() saved those values
# and joblib.load() restores them, so predict() only applies the stored
# formula to the new input.

# EXERCISE 2 - Retrain with BOTH features so attendance matters
X_both = data[["study_hours", "attendance_percentage"]]
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_both, y, test_size=0.25, random_state=42, stratify=y
)

model_v2 = LogisticRegression(max_iter=1000)
model_v2.fit(Xb_train, yb_train)
print(f"Two-feature model accuracy on test data: {accuracy_score(yb_test, model_v2.predict(Xb_test)):.2f}")

joblib.dump(model_v2, "student_pass_model_v2.joblib")
print("Saved: student_pass_model_v2.joblib")
