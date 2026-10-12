"""
train_model.py
Trains a Logistic Regression model and saves it using joblib.
"""
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Columns: [study_hours, attendance_percentage]
X_train = np.array([
    [1, 50], [2, 55], [3, 60], [3, 65], [4, 70],
    [5, 75], [6, 80], [7, 85], [8, 90], [9, 95]
])
y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X_train, y_train)
print("Model trained successfully.")

MODEL_FILENAME = "student_pass_model.joblib"
joblib.dump(model, MODEL_FILENAME)
print(f"Model saved as '{MODEL_FILENAME}'")

loaded_model = joblib.load(MODEL_FILENAME)
print("Model loaded from disk.\n")

test_cases = [(2, 55), (5, 80), (8, 92)]
print("=" * 55)
print("PREDICTIONS FROM LOADED MODEL")
print("=" * 55)
for hours, attendance in test_cases:
    prediction = loaded_model.predict([[hours, attendance]])[0]
    probability = loaded_model.predict_proba([[hours, attendance]])[0]
    label = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours:>2} | Attendance: {attendance}% -> {label} (confidence: {max(probability)*100:.2f}%)")
print("=" * 55)
print("""
WHY NO RETRAINING IS NEEDED:
The trained model's learned parameters (coefficients and intercept)
were already stored inside the .joblib file. When we call
joblib.load(), those parameters are restored exactly as they were.
Because the model already knows the pattern from training,
we can immediately use it to predict new data without
running model.fit() again.
""")