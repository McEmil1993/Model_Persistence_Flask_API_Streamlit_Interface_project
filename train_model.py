from sklearn.linear_model import LogisticRegression
import joblib

# Sample training data
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")

# Make 3 predictions
for hours in [[2], [4], [6]]:
    prediction = loaded_model.predict([hours])[0]
    print(f"Study hours: {hours[0]}, Prediction: {prediction}")