"""
train_model.py — Balanced dataset para may S-curve ang chart
"""
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Balanced training data — may FAIL at PASS sa iba't ibang levels
# Study hours, Attendance
X_train = np.array([
    # FAIL cases (low hours OR low attendance)
    [1, 50], [1, 70], [1, 90],
    [2, 50], [2, 60], [2, 85],
    [3, 55], [3, 65],
    [4, 60], [4, 70],
    [5, 65],
    # PASS cases (high hours AND high attendance)
    [5, 80], [5, 90],
    [6, 75], [6, 85], [6, 95],
    [7, 80], [7, 90],
    [8, 85], [8, 95],
    [9, 90], [10, 95]
])
y_train = np.array([
    0, 0, 0,     # Fail
    0, 0, 0,     # Fail
    0, 0,        # Fail
    0, 0,        # Fail
    0,           # Fail
    1, 1,        # Pass
    1, 1, 1,     # Pass
    1, 1,        # Pass
    1, 1,        # Pass
    1, 1         # Pass
])

model = LogisticRegression()
model.fit(X_train, y_train)
print("Model trained successfully with balanced data.")

MODEL_FILENAME = "student_pass_model.joblib"
joblib.dump(model, MODEL_FILENAME)
print(f"Model saved as '{MODEL_FILENAME}'")

loaded_model = joblib.load(MODEL_FILENAME)
print("Model loaded from disk.\n")

# 3 predictions with different study hours
test_cases = [(2, 75), (5, 75), (8, 75)]
print("=" * 65)
print("PREDICTIONS FROM LOADED MODEL")
print("=" * 65)
for hours, attendance in test_cases:
    prediction = loaded_model.predict([[hours, attendance]])[0]
    probability = loaded_model.predict_proba([[hours, attendance]])[0]
    label = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours:>2} | Attendance: {attendance}% "
          f"-> {label} (confidence: {max(probability)*100:.2f}%)")
print("=" * 65)

print("""
WHY NO RETRAINING IS NEEDED:
The trained model's learned parameters (coefficients and intercept)
were already stored inside the .joblib file. When we call
joblib.load(), those parameters are restored exactly as they were.
Because the model already knows the pattern from training,
we can immediately use it to predict new data without
running model.fit() again.
""")
