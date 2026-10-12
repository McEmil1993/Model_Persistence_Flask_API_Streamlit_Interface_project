# ============================================================
# STUDENT LASTNAME, FIRSTNAME
# ============================================================
# Exercise 1 — Model Persistence
# Replace the placeholder above with your own name before submission.

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

MODEL_FILE = "student_pass_model.joblib"

# Example training data: the model currently uses study hours only.
# Replace this sample data with your instructor's provided dataset if available.
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array(["Fail", "Fail", "Fail", "Fail", "Fail", "Pass",
              "Pass", "Pass", "Pass", "Pass", "Pass"])

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the trained model to disk.
joblib.dump(model, MODEL_FILE)
print(f"Saved trained model to {MODEL_FILE}")

# Load the saved model. No retraining is needed for prediction.
loaded_model = joblib.load(MODEL_FILE)
print("Loaded saved model successfully.")

# Make three predictions using different study-hour values.
test_study_hours = [[2], [5], [8]]
for sample in test_study_hours:
    prediction = loaded_model.predict(sample)[0]
    print(f"Study hours: {sample[0]} -> Predicted result: {prediction}")

print(
    "\nWhy retraining is not needed: joblib.load() restores the already-trained "
    "model, including its learned decision rules. It can predict new inputs "
    "directly; retraining is only needed when updating the model with new data "
    "or changing the training process."
)
