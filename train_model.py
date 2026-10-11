
from sklearn.linear_model import LogisticRegression
import joblib

# Sample student data
# 0 = Fail, 1 = Pass
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")

# Test 3 predictions
for hours in [2, 5, 8]:
    prediction = loaded_model.predict([[hours]])[0]

    if prediction == 1:
        result = "Pass"
    else:
        result = "Fail"

    print(f"Study hours: {hours}, Prediction: {result}")
