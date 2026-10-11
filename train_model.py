"""Exercise 1 (extended) — Model Persistence, now with 2 features.

Pass rule used to label the data (so both inputs matter):
  PASS if (hours >= 6 and attendance >= 50)
       or (hours >= 4 and attendance >= 85)
       or (hours >= 8 and attendance >= 40)
"""
import joblib
from sklearn.linear_model import LogisticRegression

# 1. Build training data: [study_hours, attendance_percentage] -> pass (0/1)
X_train, y_train = [], []
for hours in range(1, 11):
    for attendance in (30, 50, 70, 90):
        passed = ((hours >= 6 and attendance >= 50)
                  or (hours >= 4 and attendance >= 85)
                  or (hours >= 8 and attendance >= 40))
        X_train.append([hours, attendance])
        y_train.append(1 if passed else 0)

# 2. Train the model
model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Model trained on {len(X_train)} samples, 2 features.")

# 3. Save the model using joblib.dump() with a custom filename
joblib.dump(model, "student_pass_model.joblib")
print("Model saved as student_pass_model.joblib")

# 4. Load the model back using joblib.load()
loaded_model = joblib.load("student_pass_model.joblib")
print("Model loaded from student_pass_model.joblib")

# 5. Predictions (no re-training) — same hours, different attendance
probes = [(4, 50), (4, 95), (8, 30), (8, 95), (5.5, 1), (2, 90)]
for hours, attendance in probes:
    pred = int(loaded_model.predict([[hours, attendance]])[0])
    conf = float(max(loaded_model.predict_proba([[hours, attendance]])[0])) * 100
    result = "PASS" if pred == 1 else "FAIL"
    print(f"Hours: {hours}, Attendance: {attendance}% -> {result} ({conf:.1f}%)")
