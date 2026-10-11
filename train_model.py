
from pathlib import Path

import joblib
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------
# DEMONSTRATION TRAINING DATA
# --------------------------------------------------
# Each row contains:
# [study_hours_per_day, attendance_percentage]
#
# 0 = Fail
# 1 = Pass
#
# IMPORTANT:
# These are illustrative, rule-generated labels,
# NOT real student records or validated academic data.
#
# Demonstration assumption:
# Pass if study hours >= 5 AND attendance >= 60%.

training_data = [
    [1, 20], [2, 40], [3, 80], [4, 90], [2, 90],
    [1, 100], [3, 50], [4, 40], [4, 70], [3, 100],
    [5, 20], [5, 40], [5, 60], [5, 80], [5, 100],
    [6, 30], [6, 50], [6, 60], [6, 80], [6, 100],
    [7, 20], [7, 40], [7, 60], [7, 80], [7, 100],
    [8, 30], [8, 50], [8, 60], [8, 80], [8, 100],
    [9, 20], [9, 40], [9, 60], [9, 80], [9, 100],
    [10, 30], [10, 50], [10, 60], [10, 80], [10, 100],
]

results = [
    int(hours >= 5 and attendance >= 60)
    for hours, attendance in training_data
]

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(training_data, results)

# Save beside this Python file
MODEL_PATH = Path(__file__).with_name("student_pass_model.joblib")
joblib.dump(model, MODEL_PATH)

print("Model trained and saved successfully!")
print("Features: study hours and attendance percentage")
print("WARNING: This model uses illustrative sample data.")

# --------------------------------------------------
# LOAD SAVED MODEL AND TEST
# --------------------------------------------------
loaded_model = joblib.load(MODEL_PATH)

test_students = [
    [2, 90],   # Low study hours, high attendance
    [8, 30],   # High study hours, low attendance
    [8, 90],   # High study hours, high attendance
    [5, 60],   # Exactly at the demonstration threshold
]

print("\nDemonstration predictions:")

for hours, attendance in test_students:
    prediction = loaded_model.predict([[hours, attendance]])[0]
    result = "Pass" if prediction == 1 else "Fail"

    print(
        f"Study hours: {hours}, "
        f"Attendance: {attendance}% -> {result}"
    )

print("\nSaved model loaded and tested successfully.")
