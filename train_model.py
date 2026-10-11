
import joblib
from sklearn.tree import DecisionTreeClassifier

# 1. Create the student dataset
# Study hours and student results
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
results = [
    "Fail", "Fail", "Fail", "Fail", "Pass",
    "Pass", "Pass", "Pass", "Pass", "Pass"
]

# 2. Train the machine learning model
model = DecisionTreeClassifier(random_state=42)
model.fit(study_hours, results)

# 3. Save the trained model
joblib.dump(model, "student_pass_model.joblib")
print("Model trained and saved successfully!")

# 4. Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")
print("Saved model loaded successfully!")

# 5. Make three predictions
test_hours = [[2], [5], [8]]

for hours in test_hours:
    prediction = loaded_model.predict([hours])[0]
    print(f"Study Hours: {hours[0]} | Prediction: {prediction}")

# 6. Explain model persistence
print("\nThe trained model was saved and loaded.")
print("Retraining is not needed to make these predictions.")
