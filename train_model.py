import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("student_data.csv")

print("STUDENT PERFORMANCE DATASET")
print(data)

# Input and target
X = data[["StudyHours"]]
y = data["Result"]

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "student_pass_model.joblib")

print("\nModel trained successfully!")
print("Model saved as student_pass_model.joblib")

# Load saved model
loaded_model = joblib.load("student_pass_model.joblib")

# # Make three predictions
test_hours = [2, 5, 8]

print("\nTHREE SAMPLE PREDICTIONS")

for hours in test_hours:
    prediction = loaded_model.predict([[hours]])[0]
    print(f"Study Hours: {hours} -> Prediction: {prediction}")