
import joblib
from sklearn.linear_model import LogisticRegression

# Sample classroom dataset
# Features: study hours
X = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model trained successfully!")
print("Model saved as student_pass_model.joblib")

# Test three predictions
test_hours = [[2], [5], [8]]

for hours in test_hours:
    prediction = model.predict([hours])[0]
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours[0]} -> Prediction: {result}")
