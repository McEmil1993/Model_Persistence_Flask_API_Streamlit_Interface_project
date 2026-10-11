import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# =========================================
# 1. Data (study hours -> pass/fail)
# =========================================
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# =========================================
# 2. Train the model
# =========================================
model = LogisticRegression()
model.fit(X, y)
print("✅ Model trained!\n")


# =========================================
# ✅ TASK 1: Save the model using joblib.dump()
# =========================================
joblib.dump(model, "student_pass_model.joblib")
print("[TASK 1] ✅ Model saved using joblib.dump()")
print("         Filename: student_pass_model.joblib\n")


# =========================================
# ✅ TASK 2: Load the model using joblib.load()
# =========================================
loaded_model = joblib.load("student_pass_model.joblib")
print("[TASK 2] ✅ Model loaded using joblib.load()\n")


# =========================================
# ✅ TASK 3: Make 3 predictions using different study hours
# =========================================
print("[TASK 3] Making 3 predictions...")
test_hours = [2.5, 5.5, 9]

for hours in test_hours:
    result = loaded_model.predict([[hours]])[0]
    label = "PASS" if result == 1 else "FAIL"
    print(f"         {hours} study hours  ->  {label}")
print()


# =========================================
# ✅ TASK 4: Explain why no retraining is needed
# =========================================
print("[TASK 4] Why does the model NOT need to be trained again?")
print("-" * 60)
print("""
When we train a Logistic Regression model, it learns two numbers:
    1. The COEFFICIENT (weight) for study hours
    2. The INTERCEPT (bias)

These learned numbers are what make the model "smart".

joblib.dump()  ->  saves those learned numbers into a file
                   (student_pass_model.joblib)

joblib.load()  ->  reads those exact same numbers back into memory

Because the learned numbers are restored perfectly, the loaded model
behaves EXACTLY like the original trained model. There is no "learning"
that needs to happen again - the model already knows what it knows.

We only need to retrain if:
    - We have NEW data and want the model to learn from it, OR
    - We change the features (e.g., add attendance_percentage)
""")