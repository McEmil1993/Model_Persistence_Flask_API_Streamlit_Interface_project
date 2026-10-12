import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample training data: study hours vs pass (0 = Fail, 1 = Pass)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model using joblib.dump()
joblib.dump(model, 'student_pass_model.joblib')
print("Model saved as student_pass_model.joblib")

# Load the saved model using joblib.load()
loaded_model = joblib.load('student_pass_model.joblib')

# Make three predictions using different study hours
study_hours_list = [3, 5.5, 8]

for hours in study_hours_list:
    prediction = loaded_model.predict([[hours]])[0]
    probability = loaded_model.predict_proba([[hours]])[0][1]
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours} -> Prediction: {result} (Pass probability: {probability:.2f})")
