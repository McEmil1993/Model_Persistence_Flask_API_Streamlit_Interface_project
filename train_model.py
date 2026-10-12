import joblib
from sklearn.linear_model import LogisticRegression

# Sample training data: study hours and pass/fail result
# 0 = Fail, 1 = Pass
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
results = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Create and train the model
model = LogisticRegression()
model.fit(study_hours, results)

# Save the trained model
model_filename = "student_pass_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# Load the saved model
loaded_model = joblib.load(model_filename)
print("Saved model loaded successfully!")

# Make three predictions without retraining
test_hours = [[2], [5], [8]]

for hours in test_hours:
    prediction = loaded_model.predict([hours])[0]
    status = "Pass" if prediction == 1 else "Fail"
    print(f"Study hours: {hours[0]} | Prediction: {status}")

print("\nThe saved model can make predictions without being trained again.")