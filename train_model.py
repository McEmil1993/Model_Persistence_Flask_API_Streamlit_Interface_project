"""Exercise 1 (and the retraining note in Exercise 2).

The provided model uses only study hours. Attendance only affects the
prediction after the model is retrained with it (set USE_ATTENDANCE = True
and run this script again).
"""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MODEL_FILE = "student_pass_model.joblib"

# False = provided model (study hours only)
# True  = retrain so that attendance_percentage influences the predictions
USE_ATTENDANCE = False

FEATURES = ["study_hours", "attendance_percentage"] if USE_ATTENDANCE else ["study_hours"]

# Small sample dataset (1 = pass, 0 = fail)
data = pd.DataFrame({
    "study_hours":           [1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 1, 4, 6, 8],
    "attendance_percentage": [40, 55, 70, 60, 80, 65, 75, 90, 85, 95, 90, 98, 90, 50, 60, 70],
    "passed":                [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
})

X = data[FEATURES]
y = data["passed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Features used: {FEATURES}")
print(f"Test accuracy: {model.score(X_test, y_test):.2f}")

# --- Task 1: save the model ---
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")

# --- Task 2: load the model (no training needed from here on) ---
loaded_model = joblib.load(MODEL_FILE)

# --- Task 3: three predictions with different study hours ---
# Attendance is set to 80% for all three. It is ignored unless USE_ATTENDANCE is True.
samples = pd.DataFrame({
    "study_hours": [1, 4, 8],
    "attendance_percentage": [80, 80, 80],
})
predictions = loaded_model.predict(samples[FEATURES])
probabilities = loaded_model.predict_proba(samples[FEATURES])[:, 1]

for hours, pred, prob in zip(samples["study_hours"], predictions, probabilities):
    label = "PASS" if pred == 1 else "FAIL"
    print(f"{hours} study hours -> {label} (pass probability {prob:.2f})")

# --- Task 4: why no retraining is needed ---
print(
    "\nWhy no retraining? Training (fit) is what learns the coefficients and "
    "intercept. joblib.dump() stores those learned values, and joblib.load() "
    "restores them, so predict() just applies the saved formula to new inputs. "
    "Attendance only changes the result if the model was retrained with it."
)